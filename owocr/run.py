import sys
import signal
import time
import threading
from pathlib import Path
import queue
import io
import re
import logging
import inspect
import os
import json
from dataclasses import asdict

import numpy as np
import pyperclipfix
import mss
import psutil
import asyncio
import websockets
import socketserver

from PIL import Image, UnidentifiedImageError
from loguru import logger
from pynput import keyboard
from desktop_notifier import DesktopNotifierSync, Urgency

from .ocr import *
from .config import config
from .screen_coordinate_picker import get_screen_selection, terminate_selector_if_running

try:
    import win32gui
    import win32ui
    import win32api
    import win32con
    import win32process
    import win32clipboard
    import pywintypes
    import ctypes
except ImportError:
    pass

try:
    import objc
    import platform
    from AppKit import NSData, NSImage, NSBitmapImageRep, NSDeviceRGBColorSpace, NSGraphicsContext, NSZeroPoint, NSZeroRect, NSCompositingOperationCopy
    from Quartz import CGWindowListCreateImageFromArray, kCGWindowImageBoundsIgnoreFraming, CGRectMake, CGRectNull, CGMainDisplayID, CGWindowListCopyWindowInfo, \
                       CGWindowListCreateDescriptionFromArray, kCGWindowListOptionOnScreenOnly, kCGWindowListExcludeDesktopElements, kCGWindowName, kCGNullWindowID, \
                       CGImageGetWidth, CGImageGetHeight, CGDataProviderCopyData, CGImageGetDataProvider, CGImageGetBytesPerRow, kCGWindowImageNominalResolution
    from ScreenCaptureKit import SCContentFilter, SCScreenshotManager, SCShareableContent, SCStreamConfiguration, SCCaptureResolutionNominal
except ImportError:
    pass


class ClipboardThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.delay_secs = config.get_general('delay_secs')
        self.last_update = time.time()

    def are_images_identical(self, img1, img2):
        if None in (img1, img2):
            return img1 == img2

        img1 = np.array(img1)
        img2 = np.array(img2)

        return (img1.shape == img2.shape) and (img1 == img2).all()

    def normalize_macos_clipboard(self, img):
        ns_data = NSData.dataWithBytes_length_(img, len(img))
        ns_image = NSImage.alloc().initWithData_(ns_data)

        new_image = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
            None,  # Set to None to create a new bitmap
            int(ns_image.size().width),
            int(ns_image.size().height),
            8,  # Bits per sample
            4,  # Samples per pixel (R, G, B, A)
            True,  # Has alpha
            False,  # Is not planar
            NSDeviceRGBColorSpace,
            0,  # Automatically compute bytes per row
            32  # Bits per pixel (8 bits per sample * 4 samples per pixel)
        )

        context = NSGraphicsContext.graphicsContextWithBitmapImageRep_(new_image)
        NSGraphicsContext.setCurrentContext_(context)

        ns_image.drawAtPoint_fromRect_operation_fraction_(
            NSZeroPoint,
            NSZeroRect,
            NSCompositingOperationCopy,
            1.0
        )

        return bytes(new_image.TIFFRepresentation())

    def process_message(self, hwnd: int, msg: int, wparam: int, lparam: int):
        WM_CLIPBOARDUPDATE = 0x031D
        timestamp = time.time()
        if msg == WM_CLIPBOARDUPDATE and timestamp - self.last_update > 1 and not paused.is_set():
            self.last_update = timestamp
            while True:
                try:
                    win32clipboard.OpenClipboard()
                    break
                except pywintypes.error:
                    pass
                time.sleep(0.1)
            try:
                if win32clipboard.IsClipboardFormatAvailable(win32con.CF_BITMAP) and win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_DIB):
                    img = win32clipboard.GetClipboardData(win32clipboard.CF_DIB)
                    image_queue.put((img, False))
                win32clipboard.CloseClipboard()
            except pywintypes.error:
                pass
        return 0

    def create_window(self):
        className = 'ClipboardHook'
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = self.process_message
        wc.lpszClassName = className
        wc.hInstance = win32api.GetModuleHandle(None)
        class_atom = win32gui.RegisterClass(wc)
        return win32gui.CreateWindow(class_atom, className, 0, 0, 0, 0, 0, 0, 0, wc.hInstance, None)

    def run(self):
        if sys.platform == 'win32':
            hwnd = self.create_window()
            self.thread_id = win32api.GetCurrentThreadId()
            ctypes.windll.user32.AddClipboardFormatListener(hwnd)
            win32gui.PumpMessages()
        else:
            is_macos = sys.platform == 'darwin'
            if is_macos:
                from AppKit import NSPasteboard, NSPasteboardTypeTIFF
                pasteboard = NSPasteboard.generalPasteboard()
                count = pasteboard.changeCount()
            else:
                from PIL import ImageGrab
            process_clipboard = False
            img = None

            while not terminated.is_set():
                if paused.is_set():
                    sleep_time = 0.5
                    process_clipboard = False
                else:
                    sleep_time = self.delay_secs
                    if is_macos:
                        with objc.autorelease_pool():
                            old_count = count
                            count = pasteboard.changeCount()
                            if process_clipboard and count != old_count:
                                while len(pasteboard.types()) == 0:
                                    time.sleep(0.1)
                                if NSPasteboardTypeTIFF in pasteboard.types():
                                    img = self.normalize_macos_clipboard(pasteboard.dataForType_(NSPasteboardTypeTIFF))
                                    image_queue.put((img, False))
                    else:
                        old_img = img
                        try:
                            img = ImageGrab.grabclipboard()
                        except Exception:
                            pass
                        else:
                            if (process_clipboard and isinstance(img, Image.Image) and \
                                (not self.are_images_identical(img, old_img))):
                                image_queue.put((img, False))

                    process_clipboard = True

                if not terminated.is_set():
                    time.sleep(sleep_time)


class DirectoryWatcher(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path
        self.delay_secs = config.get_general('delay_secs')
        self.last_update = time.time()
        self.allowed_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')

    def get_path_key(self, path):
        return path, path.lstat().st_mtime

    def run(self):
        old_paths = set()
        for path in self.path.iterdir():
            if path.suffix.lower() in self.allowed_extensions:
                old_paths.add(self.get_path_key(path))

        while not terminated.is_set():
            if paused.is_set():
                sleep_time = 0.5
            else:
                sleep_time = self.delay_secs
                for path in self.path.iterdir():
                    if path.suffix.lower() in self.allowed_extensions:
                        path_key = self.get_path_key(path)
                        if path_key not in old_paths:
                            old_paths.add(path_key)

                            if not paused.is_set():
                                image_queue.put((path, False))

            if not terminated.is_set():
                time.sleep(sleep_time)


class WebsocketServerThread(threading.Thread):
    def __init__(self, read):
        super().__init__(daemon=True)
        self._loop = None
        self.read = read
        self.clients = set()
        self._event = threading.Event()

    @property
    def loop(self):
        self._event.wait()
        return self._loop

    async def send_text_coroutine(self, text):
        for client in self.clients:
            await client.send(text)

    async def server_handler(self, websocket):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                if self.read and not paused.is_set():
                    image_queue.put((message, False))
                    try:
                        await websocket.send('True')
                    except websockets.exceptions.ConnectionClosedOK:
                        pass
                else:
                    try:
                        await websocket.send('False')
                    except websockets.exceptions.ConnectionClosedOK:
                        pass
        except websockets.exceptions.ConnectionClosedError:
            pass
        finally:
            self.clients.remove(websocket)

    def send_text(self, text):
        return asyncio.run_coroutine_threadsafe(self.send_text_coroutine(text), self.loop)

    def stop_server(self):
        self.loop.call_soon_threadsafe(self._stop_event.set)

    def run(self):
        async def main():
            self._loop = asyncio.get_running_loop()
            self._stop_event = stop_event = asyncio.Event()
            self._event.set()
            self.server = start_server = websockets.serve(self.server_handler, '0.0.0.0', config.get_general('websocket_port'), max_size=1000000000)
            async with start_server:
                await stop_event.wait()
        asyncio.run(main())


class RequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        conn = self.request
        conn.settimeout(3)
        data = conn.recv(4)
        img_size = int.from_bytes(data)
        img = bytearray()
        try:
            while len(img) < img_size:
                data = conn.recv(4096)
                if not data:
                    break
                img.extend(data)
        except TimeoutError:
            pass

        if not paused.is_set():
            image_queue.put((img, False))
            conn.sendall(b'True')
        else:
            conn.sendall(b'False')


class TextFiltering:
    def __init__(self):
        self.language = config.get_general('language')
        self.frame_stabilization = 0 if config.get_general('screen_capture_delay_secs') == -1 else config.get_general('screen_capture_frame_stabilization')
        self.line_recovery = config.get_general('screen_capture_line_recovery')
        self.furigana_filter = self.language == 'ja' and config.get_general('furigana_filter')
        self.last_frame_data = (None, None)
        self.last_last_frame_data = (None, None)
        self.stable_frame_data = None
        self.last_frame_text = ([], None)
        self.last_last_frame_text = ([], None)
        self.stable_frame_text = []
        self.processed_stable_frame = False
        self.frame_stabilization_timestamp = 0
        self.cj_regex = re.compile(r'[\u3041-\u3096\u30A1-\u30FA\u4E01-\u9FFF]')
        self.kanji_regex = re.compile(r'[\u4E00-\u9FFF]')
        self.regex = self.get_regex()
        self.kana_variants = {
            'ぁ': ['ぁ', 'あ'], 'あ': ['ぁ', 'あ'],
            'ぃ': ['ぃ', 'い'], 'い': ['ぃ', 'い'],
            'ぅ': ['ぅ', 'う'], 'う': ['ぅ', 'う'],
            'ぇ': ['ぇ', 'え'], 'え': ['ぇ', 'え'],
            'ぉ': ['ぉ', 'お'], 'お': ['ぉ', 'お'],
            'ァ': ['ァ', 'ア'], 'ア': ['ァ', 'ア'],
            'ィ': ['ィ', 'イ'], 'イ': ['ィ', 'イ'],
            'ゥ': ['ゥ', 'ウ'], 'ウ': ['ゥ', 'ウ'],
            'ェ': ['ェ', 'エ'], 'エ': ['ェ', 'エ'],
            'ォ': ['ォ', 'オ'], 'オ': ['ォ', 'オ'],
            'ゃ': ['ゃ', 'や'], 'や': ['ゃ', 'や'],
            'ゅ': ['ゅ', 'ゆ'], 'ゆ': ['ゅ', 'ゆ'],
            'ょ': ['ょ', 'よ'], 'よ': ['ょ', 'よ'],
            'ャ': ['ャ', 'ヤ'], 'ヤ': ['ャ', 'ヤ'],
            'ュ': ['ュ', 'ユ'], 'ユ': ['ュ', 'ユ'],
            'ョ': ['ョ', 'ヨ'], 'ヨ': ['ョ', 'ヨ'],
            'っ': ['っ', 'つ'], 'つ': ['っ', 'つ'],
            'ッ': ['ッ', 'ツ'], 'ツ': ['ッ', 'ツ'],
            'ゎ': ['ゎ', 'わ'], 'わ': ['ゎ', 'わ'],
            'ヮ': ['ヮ', 'ワ'], 'ワ': ['ヮ', 'ワ']
        }

    def get_regex(self):
        if self.language == 'ja':
            return self.cj_regex
        elif self.language == 'zh':
            return self.kanji_regex
        elif self.language == 'ko':
            return re.compile(r'[\uAC00-\uD7AF]')
        elif self.language == 'ar':
            return re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
        elif self.language == 'ru':
            return re.compile(r'[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F\u1C80-\u1C8F]')
        elif self.language == 'el':
            return re.compile(r'[\u0370-\u03FF\u1F00-\u1FFF]')
        elif self.language == 'he':
            return re.compile(r'[\u0590-\u05FF\uFB1D-\uFB4F]')
        elif self.language == 'th':
            return re.compile(r'[\u0E00-\u0E7F]')
        else:
            # Latin Extended regex for many European languages/English
            return re.compile(
            r'[a-zA-Z\u00C0-\u00FF\u0100-\u017F\u0180-\u024F\u0250-\u02AF\u1D00-\u1D7F\u1D80-\u1DBF\u1E00-\u1EFF\u2C60-\u2C7F\uA720-\uA7FF\uAB30-\uAB6F]')

    def convert_small_kana_to_big(self, text):
        converted_text = ''.join(self.kana_variants.get(char, [char])[-1] for char in text)
        return converted_text

    def _get_line_text(self, line):
        if line.text is not None:
            return line.text
        text_parts = []
        for w in line.words:
            text_parts.append(w.text)
            if w.separator is not None:
                text_parts.append(w.separator)
            else:
                text_parts.append(' ')
        return ''.join(text_parts)

    def _normalize_line_for_comparison(self, line_text):
        if not line_text:
            return ''
        filtered_text = ''.join(self.regex.findall(line_text))
        if self.language == 'ja':
            filtered_text = self.convert_small_kana_to_big(filtered_text)
        return filtered_text

    def _find_changed_lines(self, pil_image, current_result):
        if self.frame_stabilization == 0:
            changed_lines = self._find_changed_lines_impl(current_result, self.last_frame_data[1])
            if changed_lines == None:
                return 0, 0, None
            changed_lines_count = len(changed_lines)
            self.last_frame_data = (pil_image, current_result)
            if changed_lines_count and config.get_general('output_format') != 'json':
                changed_regions_image = self._create_changed_regions_image(pil_image, changed_lines, None, None)
                if not changed_regions_image:
                    logger.warning('Error occurred while creating the differential image.')
                    return 0, 0, None
                return changed_lines_count, 0, changed_regions_image
            else:
                return changed_lines_count, 0, None

        changed_lines_stabilization = self._find_changed_lines_impl(current_result, self.last_frame_data[1])
        if changed_lines_stabilization == None:
            return 0, 0, None

        frames_match = len(changed_lines_stabilization) == 0

        logger.debug(f"Frames match: '{frames_match}'")

        if frames_match:
            if self.processed_stable_frame:
                return 0, 0, None
            if time.time() - self.frame_stabilization_timestamp < self.frame_stabilization:
                return 0, 0, None
            changed_lines = self._find_changed_lines_impl(current_result, self.stable_frame_data)
            if self.line_recovery and self.last_last_frame_data:
                logger.debug(f'Checking for missed lines')
                recovered_lines = self._find_changed_lines_impl(self.last_last_frame_data[1], self.stable_frame_data, current_result)
                recovered_lines_count = len(recovered_lines) if recovered_lines else 0
            else:
                recovered_lines_count = 0
                recovered_lines = []
            self.processed_stable_frame = True
            self.stable_frame_data = current_result
            changed_lines_count = len(changed_lines)
            if (changed_lines_count or recovered_lines_count) and config.get_general('output_format') != 'json':
                if recovered_lines:
                    changed_regions_image = self._create_changed_regions_image(pil_image, changed_lines, self.last_last_frame_data[0], recovered_lines)
                else:
                    changed_regions_image = self._create_changed_regions_image(pil_image, changed_lines, None, None)

                if not changed_regions_image:
                    logger.warning('Error occurred while creating the differential image.')
                    return 0, 0, None
                return changed_lines_count, recovered_lines_count, changed_regions_image
            else:
                return changed_lines_count, recovered_lines_count, None
        else:
            self.last_last_frame_data = self.last_frame_data
            self.last_frame_data = (pil_image, current_result)
            self.processed_stable_frame = False
            self.frame_stabilization_timestamp = time.time()
            return 0, 0, None

    def _find_changed_lines_impl(self, current_result, previous_result, next_result=None):
        if not current_result:
            return None

        changed_lines = []
        current_lines = []
        previous_lines = []
        current_text = []
        previous_text = []

        for p in current_result.paragraphs:
            current_lines.extend(p.lines)
        if len(current_lines) == 0:
            return None

        for current_line in current_lines:
            current_text_line = self._get_line_text(current_line)
            current_text_line = self._normalize_line_for_comparison(current_text_line)
            current_text.append(current_text_line)
        if all(not current_text_line for current_text_line in current_lines):
            return None

        if previous_result:
            for p in previous_result.paragraphs:
                previous_lines.extend(p.lines)
            if next_result:
                for p in next_result.paragraphs:
                    previous_lines.extend(p.lines)

            for previous_line in previous_lines:
                previous_text_line = self._get_line_text(previous_line)
                previous_text_line = self._normalize_line_for_comparison(previous_text_line)
                previous_text.append(previous_text_line)

        all_previous_text = ''.join(previous_text)

        logger.debug(f"Previous text: '{previous_text}'")

        for i, current_text_line in enumerate(current_text):
            if not current_text_line:
                continue

            if not next_result and len(current_text_line) < 3:
                text_similar = current_text_line in previous_text
            else:
                text_similar = current_text_line in all_previous_text

            logger.debug(f"Current line: '{current_text_line}' Similar: '{text_similar}'")

            if not text_similar:
                if next_result:
                    logger.opt(colors=True).debug(f"<red>Recovered line: '{current_text_line}'</red>")
                changed_lines.append(current_lines[i])

        return changed_lines

    def _find_changed_lines_text(self, current_result, current_result_ocr, two_pass_processing_active, recovered_lines_count):
        frame_stabilization_active = self.frame_stabilization != 0

        if (not frame_stabilization_active) or two_pass_processing_active:
            changed_lines = self._find_changed_lines_text_impl(current_result, current_result_ocr, None, self.last_frame_text[0], None, True, recovered_lines_count)
            if changed_lines == None:
                return []
            self.last_frame_text = (current_result, current_result_ocr)
            return changed_lines

        changed_lines_stabilization = self._find_changed_lines_text_impl(current_result, current_result_ocr, None, self.last_frame_text[0], None, False, 0)
        if changed_lines_stabilization == None:
            return []

        frames_match = len(changed_lines_stabilization) == 0

        logger.debug(f"Frames match: '{frames_match}'")

        if frames_match:
            if self.processed_stable_frame:
                return []
            if time.time() - self.frame_stabilization_timestamp < self.frame_stabilization:
                return []
            if self.line_recovery and self.last_last_frame_text[0]:
                logger.debug(f'Checking for missed lines')
                recovered_lines = self._find_changed_lines_text_impl(self.last_last_frame_text[0], self.last_last_frame_text[1], None, self.stable_frame_text, current_result, False, 0)
                recovered_lines_count = len(recovered_lines) if recovered_lines else 0
            else:
                recovered_lines_count = 0
                recovered_lines = []
            changed_lines = self._find_changed_lines_text_impl(current_result, current_result_ocr, recovered_lines, self.stable_frame_text, None, True, recovered_lines_count)
            self.processed_stable_frame = True
            self.stable_frame_text = current_result
            return changed_lines
        else:
            self.last_last_frame_text = self.last_frame_text
            self.last_frame_text = (current_result, current_result_ocr)
            self.processed_stable_frame = False
            self.frame_stabilization_timestamp = time.time()
            return []

    def _find_changed_lines_text_impl(self, current_result, current_result_ocr, recovered_lines, previous_result, next_result, filtering, recovered_lines_count):
        if recovered_lines:
            current_result = recovered_lines + current_result

        if len(current_result) == 0:
            return None

        changed_lines = []
        current_lines = []
        current_lines_ocr = []
        previous_text = []

        for current_line in current_result:
            current_text_line = self._normalize_line_for_comparison(current_line)
            current_lines.append(current_text_line)
        if all(not current_text_line for current_text_line in current_lines):
            return None

        if self.furigana_filter and isinstance(current_result_ocr, OcrResult):
            for p in current_result_ocr.paragraphs:
                current_lines_ocr.extend(p.lines)

        for prev_line in previous_result:
            prev_text = self._normalize_line_for_comparison(prev_line)
            previous_text.append(prev_text)
        if next_result != None:
            for next_text in next_result:
                previous_text.extend(next_text)

        all_previous_text = ''.join(previous_text)

        logger.opt(colors=True).debug(f"<magenta>Previous text: '{previous_text}'</magenta>")

        first = True
        for i, current_text in enumerate(current_lines):
            if not current_text:
                continue

            if next_result != None and len(current_text) < 3:
                text_similar = current_text in previous_text
            else:
                text_similar = current_text in all_previous_text

            logger.opt(colors=True).debug(f"<magenta>Current line: '{current_text}' Similar: '{text_similar}'</magenta>")

            if text_similar:
                continue

            if recovered_lines_count > 0:
                if any(line.startswith(current_text) for j, line in enumerate(current_lines) if i != j):
                    logger.opt(colors=True).debug(f"<magenta>Skipping recovered line: '{current_text}'</magenta>")
                    recovered_lines_count -= 1
                    continue

            changed_line = current_result[i]

            if next_result != None:
                logger.opt(colors=True).debug(f"<red>Recovered line: '{changed_line}'</red>")

            if current_lines_ocr:
                i2 = i if not recovered_lines else i - len(recovered_lines)
                if i2 >= 0:
                    current_line_bbox = current_lines_ocr[i2].bounding_box

                    # Check if line contains only kana (no kanji)
                    has_kanji = self.kanji_regex.search(current_text)

                    if not has_kanji:
                        is_furigana = False

                        for j in range(len(current_lines_ocr)):
                            if i2 == j:
                                continue
                            if not current_lines[j]:
                                continue

                            other_line_bbox = current_lines_ocr[j].bounding_box
                            other_line_text = current_lines[j]

                            if len(current_text) <= len(other_line_text):
                                is_vertical = other_line_bbox.height > other_line_bbox.width
                            else:
                                is_vertical = current_line_bbox.height > current_line_bbox.width

                            logger.opt(colors=True).debug(f"<magenta>Furigana check against line: '{other_line_text}'</magenta>")

                            if is_vertical:
                                width_threshold = other_line_bbox.width * 0.7
                                is_smaller = current_line_bbox.width < width_threshold
                                logger.opt(colors=True).debug(f"<magenta>Vertical furigana check width: '{other_line_bbox.width}' '{current_line_bbox.width}'</magenta>")
                            else:
                                height_threshold = other_line_bbox.height * 0.7
                                is_smaller = current_line_bbox.height < height_threshold
                                logger.opt(colors=True).debug(f"<magenta>Horizontal furigana check height: '{other_line_bbox.height}' '{current_line_bbox.height}'</magenta>")

                            if not is_smaller:
                                continue

                            # Check if the line has kanji
                            other_has_kanji = self.kanji_regex.search(other_line_text)
                            if not other_has_kanji:
                                continue

                            if is_vertical:
                                horizontal_threshold = current_line_bbox.width + other_line_bbox.width
                                horizontal_distance = current_line_bbox.center_x - other_line_bbox.center_x
                                vertical_overlap = self._check_vertical_overlap(current_line_bbox, other_line_bbox)

                                logger.opt(colors=True).debug(f"<magenta>Vertical furigana check position: '{horizontal_threshold}' '{horizontal_distance}' '{vertical_overlap}'</magenta>")

                                # If horizontally close and vertically aligned, it's likely furigana
                                if (0 < horizontal_distance < horizontal_threshold and vertical_overlap > 0.5):
                                    is_furigana = True
                                    logger.opt(colors=True).debug(f"<magenta>Skipping vertical furigana line: '{current_text}' next to line: '{other_line_text}'</magenta>")
                                    break
                            else:
                                vertical_threshold = other_line_bbox.height + current_line_bbox.height
                                vertical_distance = other_line_bbox.center_y - current_line_bbox.center_y
                                horizontal_overlap = self._check_horizontal_overlap(current_line_bbox, other_line_bbox)

                                logger.opt(colors=True).debug(f"<magenta>Horizontal furigana check position: '{vertical_threshold}' '{vertical_distance}' '{horizontal_overlap}'</magenta>")

                                # If vertically close and horizontally aligned, it's likely furigana
                                if (0 < vertical_distance < vertical_threshold and horizontal_overlap > 0.5):
                                    is_furigana = True
                                    logger.opt(colors=True).debug(f"<magenta>Skipping horizontal furigana line: '{current_text}' above line: '{other_line_text}'</magenta>")
                                    break

                        if is_furigana:
                            continue

            if first and len(current_text) > 3:
                first = False
                # For the first line, check if it contains the end of previous text
                if filtering and all_previous_text:
                    overlap = self._find_overlap(all_previous_text, current_text)
                    if overlap and len(current_text) > len(overlap):
                        logger.opt(colors=True).debug(f"<magenta>Found overlap: '{overlap}'</magenta>")
                        changed_line = self._cut_at_overlap(changed_line, overlap)
                        logger.opt(colors=True).debug(f"<magenta>After cutting: '{changed_line}'</magenta>")
            changed_lines.append(changed_line)

        return changed_lines

    def _standalone_furigana_filter(self, result, result_ocr):
        return self._find_changed_lines_text_impl(result, result_ocr, None, [], None, False, 0)

    def _find_overlap(self, previous_text, current_text):
        min_overlap_length = 3
        max_overlap_length = min(len(previous_text), len(current_text))

        for overlap_length in range(max_overlap_length, min_overlap_length - 1, -1):
            previous_end = previous_text[-overlap_length:]
            current_start = current_text[:overlap_length]

            if previous_end == current_start:
                return previous_end

        return None

    def _cut_at_overlap(self, current_line, overlap):
        pattern_parts = []
        for char in overlap:
            if char in self.kana_variants:
                variants = self.kana_variants[char]
                pattern_parts.append(f'[{"".join(variants)}]')
            else:
                pattern_parts.append(re.escape(char))

        overlap_pattern = r'.*?'.join(pattern_parts)
        full_pattern = r'^.*?' + overlap_pattern

        logger.opt(colors=True).debug(f"<magenta>Cut regex: '{full_pattern}'</magenta>")

        match = re.search(full_pattern, current_line)
        if match:
            cut_position = match.end()
            return current_line[cut_position:]

        return current_line

    def _check_horizontal_overlap(self, bbox1, bbox2):
        # Calculate left and right boundaries for both boxes
        left1 = bbox1.center_x - bbox1.width / 2
        right1 = bbox1.center_x + bbox1.width / 2
        left2 = bbox2.center_x - bbox2.width / 2
        right2 = bbox2.center_x + bbox2.width / 2

        # Calculate overlap
        overlap_left = max(left1, left2)
        overlap_right = min(right1, right2)

        if overlap_right <= overlap_left:
            return 0.0

        overlap_width = overlap_right - overlap_left
        smaller_width = min(bbox1.width, bbox2.width)

        return overlap_width / smaller_width if smaller_width > 0 else 0.0

    def _check_vertical_overlap(self, bbox1, bbox2):
        # Calculate top and bottom boundaries for both boxes
        top1 = bbox1.center_y - bbox1.height / 2
        bottom1 = bbox1.center_y + bbox1.height / 2
        top2 = bbox2.center_y - bbox2.height / 2
        bottom2 = bbox2.center_y + bbox2.height / 2

        # Calculate overlap
        overlap_top = max(top1, top2)
        overlap_bottom = min(bottom1, bottom2)

        if overlap_bottom <= overlap_top:
            return 0.0

        overlap_height = overlap_bottom - overlap_top
        smaller_height = min(bbox1.height, bbox2.height)

        return overlap_height / smaller_height if smaller_height > 0 else 0.0

    def _create_changed_regions_image(self, pil_image, changed_lines, pil_image_2, changed_lines_2, margin=5):
        def crop_image(image, lines):
            img_width, img_height = image.size

            regions = []
            for line in lines:
                bbox = line.bounding_box
                x1 = (bbox.center_x - bbox.width/2) * img_width - margin
                y1 = (bbox.center_y - bbox.height/2) * img_height - margin
                x2 = (bbox.center_x + bbox.width/2) * img_width + margin
                y2 = (bbox.center_y + bbox.height/2) * img_height + margin

                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(img_width, int(x2))
                y2 = min(img_height, int(y2))

                if x2 > x1 and y2 > y1:
                    regions.append((x1, y1, x2, y2))

            if not regions:
                return None

            overall_x1 = min(x1 for x1, y1, x2, y2 in regions)
            overall_y1 = min(y1 for x1, y1, x2, y2 in regions)
            overall_x2 = max(x2 for x1, y1, x2, y2 in regions)
            overall_y2 = max(y2 for x1, y1, x2, y2 in regions)

            return image.crop((overall_x1, overall_y1, overall_x2, overall_y2))

        # Handle the case where changed_lines is empty and previous_result is provided
        if (not pil_image) and pil_image_2:
            cropped_2 = crop_image(pil_image_2, changed_lines_2)
            return cropped_2

        # Handle the case where both current and previous results are present
        elif pil_image and pil_image_2:
            # Crop both images
            cropped_1 = crop_image(pil_image, changed_lines)
            cropped_2 = crop_image(pil_image_2, changed_lines_2)

            if cropped_1 is None and cropped_2 is None:
                return None
            elif cropped_1 is None:
                return cropped_2
            elif cropped_2 is None:
                return cropped_1

            # Stitch vertically with previous_result on top
            total_width = max(cropped_1.width, cropped_2.width)
            total_height = cropped_1.height + cropped_2.height

            # Create a new image with white background
            stitched_image = Image.new('RGB', (total_width, total_height), 'white')

            # Paste previous (top) and current (bottom) images, centered horizontally
            prev_x_offset = (total_width - cropped_2.width) // 2
            stitched_image.paste(cropped_2, (prev_x_offset, 0))

            curr_x_offset = (total_width - cropped_1.width) // 2
            stitched_image.paste(cropped_1, (curr_x_offset, cropped_2.height))

            return stitched_image
        elif pil_image:
            return crop_image(pil_image, changed_lines)
        else:
            return None


class ScreenshotThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        screen_capture_area = config.get_general('screen_capture_area')
        self.coordinate_selector_combo_enabled = config.get_general('coordinate_selector_combo') != ''
        self.macos_window_tracker_instance = None
        self.windows_window_tracker_instance = None
        self.screencapture_window_active = True
        self.screencapture_window_visible = True
        if screen_capture_area == '':
            self.screencapture_mode = 0
        elif screen_capture_area.startswith('screen_'):
            parts = screen_capture_area.split('_')
            if len(parts) != 2 or not parts[1].isdigit():
                logger.error('Invalid screen_capture_area')
                sys.exit(1)
            screen_capture_monitor = int(parts[1])
            self.screencapture_mode = 1
        elif len(screen_capture_area.split(',')) == 4:
            self.screencapture_mode = 3
        else:
            self.screencapture_mode = 2

        if self.coordinate_selector_combo_enabled:
            self.launch_coordinate_picker(True, False)

        if self.screencapture_mode != 2:
            self.sct = mss.mss()

            if self.screencapture_mode == 1:
                mon = self.sct.monitors
                if len(mon) <= screen_capture_monitor:
                    logger.error('Invalid monitor number in screen_capture_area')
                    sys.exit(1)
                coord_left = mon[screen_capture_monitor]['left']
                coord_top = mon[screen_capture_monitor]['top']
                coord_width = mon[screen_capture_monitor]['width']
                coord_height = mon[screen_capture_monitor]['height']
            elif self.screencapture_mode == 3:
                coord_left, coord_top, coord_width, coord_height = [int(c.strip()) for c in screen_capture_area.split(',')]
            else:
                self.launch_coordinate_picker(False, True)

            if self.screencapture_mode != 0:
                self.sct_params = {'top': coord_top, 'left': coord_left, 'width': coord_width, 'height': coord_height}
                logger.info(f'Selected coordinates: {coord_left},{coord_top},{coord_width},{coord_height}')
        else:
            self.screen_capture_only_active_windows = config.get_general('screen_capture_only_active_windows')
            self.window_area_coordinates = None
            area_invalid_error = '"screen_capture_area" must be empty, "screen_N" where N is a screen number starting from 1, a valid set of coordinates, or a valid window name'

            if sys.platform == 'darwin':
                if config.get_general('screen_capture_old_macos_api') or int(platform.mac_ver()[0].split('.')[0]) < 14:
                    self.old_macos_screenshot_api = True
                else:
                    self.old_macos_screenshot_api = False
                    self.screencapturekit_queue = queue.Queue()
                    CGMainDisplayID()
                window_list = CGWindowListCopyWindowInfo(kCGWindowListExcludeDesktopElements, kCGNullWindowID)
                window_titles = []
                window_ids = []
                window_index = None
                for i, window in enumerate(window_list):
                    window_title = window.get(kCGWindowName, '')
                    if psutil.Process(window['kCGWindowOwnerPID']).name() not in ('Terminal', 'iTerm2'):
                        window_titles.append(window_title)
                        window_ids.append(window['kCGWindowNumber'])

                if screen_capture_area in window_titles:
                    window_index = window_titles.index(screen_capture_area)
                else:
                    for t in window_titles:
                        if screen_capture_area in t:
                            window_index = window_titles.index(t)
                            break

                if not window_index:
                    logger.error(area_invalid_error)
                    sys.exit(1)

                self.window_id = window_ids[window_index]
                window_title = window_titles[window_index]

                if self.screen_capture_only_active_windows:
                    self.macos_window_tracker_instance = threading.Thread(target=self.macos_window_tracker)
                    self.macos_window_tracker_instance.start()
                logger.info(f'Selected window: {window_title}')
            elif sys.platform == 'win32':
                self.window_handle, window_title = self.get_windows_window_handle(screen_capture_area)

                if not self.window_handle:
                    logger.error(area_invalid_error)
                    sys.exit(1)

                ctypes.windll.shcore.SetProcessDpiAwareness(1)

                self.windows_window_tracker_instance = threading.Thread(target=self.windows_window_tracker)
                self.windows_window_tracker_instance.start()
                logger.info(f'Selected window: {window_title}')
            else:
                logger.error('Window capture is only currently supported on Windows and macOS')
                sys.exit(1)

            screen_capture_window_area = config.get_general('screen_capture_window_area')
            if screen_capture_window_area != 'window':    
                if len(screen_capture_window_area.split(',')) == 4:
                    x, y, x2, y2 = [int(c.strip()) for c in screen_capture_window_area.split(',')]
                    logger.info(f'Selected window coordinates: {x},{y},{x2},{y2}')
                    self.window_area_coordinates = (img.size, (x, y, x2, y2))
                elif screen_capture_window_area == '':
                    self.launch_coordinate_picker(False, False)
                else:
                    logger.error('"screen_capture_window_area" must be empty, "window" for the whole window, or a valid set of coordinates')
                    sys.exit(1)

    def get_windows_window_handle(self, window_title):
        def callback(hwnd, window_title_part):
            window_title = win32gui.GetWindowText(hwnd)
            if window_title_part in window_title:
                handles.append((hwnd, window_title))
            return True

        handle = win32gui.FindWindow(None, window_title)
        if handle:
            return (handle, window_title)

        handles = []
        win32gui.EnumWindows(callback, window_title)
        for handle in handles:
            _, pid = win32process.GetWindowThreadProcessId(handle[0])
            if psutil.Process(pid).name().lower() not in ('cmd.exe', 'powershell.exe', 'windowsterminal.exe'):
                return handle

        return (None, None)

    def windows_window_tracker(self):
        found = True
        while not terminated.is_set():
            found = win32gui.IsWindow(self.window_handle)
            if not found:
                break
            if self.screen_capture_only_active_windows:
                self.screencapture_window_active = self.window_handle == win32gui.GetForegroundWindow()
            else:
                self.screencapture_window_visible = not win32gui.IsIconic(self.window_handle)
            time.sleep(0.5)
        if not found:
            on_window_closed(False)

    def capture_macos_window_screenshot(self, window_id):
        def shareable_content_completion_handler(shareable_content, error):
            if error:
                self.screencapturekit_queue.put(None)
                return

            target_window = None
            for window in shareable_content.windows():
                if window.windowID() == window_id:
                    target_window = window
                    break

            if not target_window:
                self.screencapturekit_queue.put(None)
                return

            with objc.autorelease_pool():
                content_filter = SCContentFilter.alloc().initWithDesktopIndependentWindow_(target_window)

                frame = content_filter.contentRect()
                width = frame.size.width
                height = frame.size.height
                configuration = SCStreamConfiguration.alloc().init()
                configuration.setSourceRect_(CGRectMake(0, 0, width, height))
                configuration.setWidth_(width)
                configuration.setHeight_(height)
                configuration.setShowsCursor_(False)
                configuration.setCaptureResolution_(SCCaptureResolutionNominal)
                configuration.setIgnoreGlobalClipSingleWindow_(True)

                SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(
                    content_filter, configuration, capture_image_completion_handler
                )

        def capture_image_completion_handler(image, error):
            if error:
                self.screencapturekit_queue.put(None)
                return

            self.screencapturekit_queue.put(image)

        SCShareableContent.getShareableContentWithCompletionHandler_(
            shareable_content_completion_handler
        )

    def macos_window_tracker(self):
        found = True
        while found and not terminated.is_set():
            found = False
            is_active = False
            with objc.autorelease_pool():
                window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
                for i, window in enumerate(window_list):
                    if found and window.get(kCGWindowName, '') == 'Fullscreen Backdrop':
                        is_active = True
                        break
                    if self.window_id == window['kCGWindowNumber']:
                        found = True
                        if i == 0 or window_list[i-1].get(kCGWindowName, '') in ('Dock', 'Color Enforcer Window'):
                            is_active = True
                            break
                if not found:
                    window_list = CGWindowListCreateDescriptionFromArray([self.window_id])
                    if len(window_list) > 0:
                        found = True
            if found:
                self.screencapture_window_active = is_active
            time.sleep(0.5)
        if not found:
            on_window_closed(False)

    def take_screenshot(self):
        if self.screencapture_mode == 2:
            if sys.platform == 'darwin':
                with objc.autorelease_pool():
                    if self.old_macos_screenshot_api:
                        cg_image = CGWindowListCreateImageFromArray(CGRectNull, [self.window_id], kCGWindowImageBoundsIgnoreFraming | kCGWindowImageNominalResolution)
                    else:
                        self.capture_macos_window_screenshot(self.window_id)
                        try:
                            cg_image = self.screencapturekit_queue.get(timeout=0.5)
                        except queue.Empty:
                            cg_image = None
                    if not cg_image:
                        return None
                    width = CGImageGetWidth(cg_image)
                    height = CGImageGetHeight(cg_image)
                    raw_data = CGDataProviderCopyData(CGImageGetDataProvider(cg_image))
                    bpr = CGImageGetBytesPerRow(cg_image)
                img = Image.frombuffer('RGBA', (width, height), raw_data, 'raw', 'BGRA', bpr, 1)
            else:
                try:
                    coord_left, coord_top, right, bottom = win32gui.GetWindowRect(self.window_handle)
                    coord_width = right - coord_left
                    coord_height = bottom - coord_top

                    hwnd_dc = win32gui.GetWindowDC(self.window_handle)
                    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
                    save_dc = mfc_dc.CreateCompatibleDC()

                    save_bitmap = win32ui.CreateBitmap()
                    save_bitmap.CreateCompatibleBitmap(mfc_dc, coord_width, coord_height)
                    save_dc.SelectObject(save_bitmap)

                    result = ctypes.windll.user32.PrintWindow(self.window_handle, save_dc.GetSafeHdc(), 2)

                    bmpinfo = save_bitmap.GetInfo()
                    bmpstr = save_bitmap.GetBitmapBits(True)
                except pywintypes.error:
                    return None
                img = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)
                try:
                    win32gui.DeleteObject(save_bitmap.GetHandle())
                except:
                    pass
                try:
                    save_dc.DeleteDC()
                except:
                    pass
                try:
                    mfc_dc.DeleteDC()
                except:
                    pass
                try:
                    win32gui.ReleaseDC(self.window_handle, hwnd_dc)
                except:
                    pass
            if self.window_area_coordinates:
                if img.size != self.window_area_coordinates[0]:
                    self.window_area_coordinates = None
                    logger.warning('Window size changed, discarding area selection')
                else:
                    img = img.crop(self.window_area_coordinates[1])
        else:
            sct_img = self.sct.grab(self.sct_params)
            img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')

        return img

    def write_result(self, result, is_combo):
        if is_combo:
            image_queue.put((result, True))
        else:
            periodic_screenshot_queue.put(result)

    def launch_coordinate_picker(self, init, must_return):
        if init:
            logger.info('Preloading coordinate picker')
            get_screen_selection(True, True)
            return
        if self.screencapture_mode != 2:
            logger.info('Launching screen coordinate picker')
            screen_selection = get_screen_selection(None, self.coordinate_selector_combo_enabled)
            if not screen_selection:
                if on_init:
                    logger.error('Picker window was closed or an error occurred')
                    sys.exit(1)
                else:
                    logger.warning('Picker window was closed or an error occurred, leaving settings unchanged')
                    return
            screen_capture_monitor = screen_selection['monitor']
            x, y, coord_width, coord_height = screen_selection['coordinates']
            if coord_width > 0 and coord_height > 0:
                coord_top = screen_capture_monitor['top'] + y
                coord_left = screen_capture_monitor['left'] + x
            else:
                logger.info('Selection is empty, selecting whole screen')
                coord_left = screen_capture_monitor['left']
                coord_top = screen_capture_monitor['top']
                coord_width = screen_capture_monitor['width']
                coord_height = screen_capture_monitor['height']
            self.sct_params = {'top': coord_top, 'left': coord_left, 'width': coord_width, 'height': coord_height}
            logger.info(f'Selected coordinates: {coord_left},{coord_top},{coord_width},{coord_height}')
        else:
            self.window_area_coordinates = None
            img = self.take_screenshot()
            logger.info('Launching window coordinate picker')
            window_selection = get_screen_selection(img, self.coordinate_selector_combo_enabled)
            if not window_selection:
                logger.warning('Picker window was closed or an error occurred, selecting whole window')
            else:
                x, y, coord_width, coord_height = window_selection['coordinates']
                if coord_width > 0 and coord_height > 0:
                    x2 = x + coord_width
                    y2 = y + coord_height
                    logger.info(f'Selected window coordinates: {x},{y},{x2},{y2}')
                    self.window_area_coordinates = (img.size, (x, y, x2, y2))
                else:
                    logger.info('Selection is empty, selecting whole window')

    def run(self):
        if self.screencapture_mode != 2:
            self.sct = mss.mss()
        while not terminated.is_set():
            if coordinate_selector_event.is_set():
                self.launch_coordinate_picker(False, False)
                coordinate_selector_event.clear()

            try:
                is_combo = screenshot_request_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            img = self.take_screenshot()
            if not img:
                self.write_result(0, is_combo)
                break

            self.write_result(img, is_combo)

        if self.macos_window_tracker_instance:
            self.macos_window_tracker_instance.join()
        elif self.windows_window_tracker_instance:
            self.windows_window_tracker_instance.join()


class AutopauseTimer:
    def __init__(self):
        self.timeout = config.get_general('auto_pause')
        self.timer_thread = threading.Thread(target=self._countdown, daemon=True)
        self.running = True
        self.countdown_active = threading.Event()
        self.allow_auto_pause = threading.Event()
        self.seconds_remaining = 0
        self.lock = threading.Lock()
        self.timer_thread.start()

    def start_timer(self):
        with self.lock:
            self.seconds_remaining = self.timeout
        self.allow_auto_pause.set()
        self.countdown_active.set()

    def stop_timer(self):
        self.countdown_active.clear()
        self.allow_auto_pause.set()

    def stop(self):
        self.running = False
        self.allow_auto_pause.set()
        self.countdown_active.set()
        if self.timer_thread.is_alive():
            self.timer_thread.join()

    def _countdown(self):
        while self.running:
            self.countdown_active.wait()
            if not self.running:
                break

            while self.running and self.countdown_active.is_set() and self.seconds_remaining > 0:
                time.sleep(1)
                with self.lock:
                    self.seconds_remaining -= 1

            self.allow_auto_pause.wait()

            if self.running and self.countdown_active.is_set() and self.seconds_remaining == 0:
                self.countdown_active.clear()
                if not (paused.is_set() or terminated.is_set()):
                    pause_handler(True)


class SecondPassThread:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.ocr_thread = None
        self.running = False

    def start(self):
        if self.ocr_thread is None or not self.ocr_thread.is_alive():
            self.running = True
            self.ocr_thread = threading.Thread(target=self._process_ocr, daemon=True)
            self.ocr_thread.start()
    
    def stop(self):
        self.running = False
        if self.ocr_thread and self.ocr_thread.is_alive():
            self.ocr_thread.join()
        while not self.input_queue.empty():
            self.input_queue.get()
        while not self.output_queue.empty():
            self.output_queue.get()

    def _process_ocr(self):
        while self.running:
            try:
                img, engine_instance, recovered_lines_count = self.input_queue.get(timeout=0.5)

                start_time = time.time()
                res, result_data = engine_instance(img)
                end_time = time.time()

                self.output_queue.put((engine_instance.readable_name, res, result_data, end_time - start_time, recovered_lines_count))
            except queue.Empty:
                continue

    def submit_task(self, img, engine_instance, recovered_lines_count):
        self.input_queue.put((img, engine_instance, recovered_lines_count))
    
    def get_result(self):
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None


class OutputResult:
    def __init__(self):
        self.screen_capture_periodic = config.get_general('screen_capture_delay_secs') != -1
        self.filtering = TextFiltering()
        self.second_pass_thread = SecondPassThread()

    def _post_process(self, text, strip_spaces):
        is_cj_text = self.filtering.cj_regex.search(''.join(text))
        line_separator = '' if strip_spaces else ' '
        if is_cj_text:
            text = line_separator.join([''.join(i.split()) for i in text])
        else:
            text = line_separator.join([re.sub(r'\s+', ' ', i).strip() for i in text])
        text = text.replace('…', '...')
        text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
        if is_cj_text:
            text = jaconv.h2z(text, ascii=True, digit=True)
        return text

    def _extract_lines_from_result(self, result_data):
        lines = []
        for p in result_data.paragraphs:
            for l in p.lines:
                lines.append(self.filtering._get_line_text(l))
        return lines

    def __call__(self, img_or_path, filter_text, auto_pause, notify):
        engine_index_local = engine_index
        output_format = config.get_general('output_format')
        engine_color = config.get_general('engine_color')
        engine_instance = engine_instances[engine_index]
        two_pass_processing_active = False
        result_data = None

        if filter_text and self.screen_capture_periodic:
            if engine_index_2 != -1 and engine_index_2 != engine_index_local and engine_instance.threading_support:
                two_pass_processing_active = True
                engine_instance_2 = engine_instances[engine_index_2]
                start_time = time.time()
                res2, result_data_2 = engine_instance_2(img_or_path)
                end_time = time.time()

                if not res2:
                    logger.opt(colors=True).warning(f'<{engine_color}>{engine_instance_2.readable_name}</{engine_color}> reported an error after {end_time - start_time:0.03f}s: {result_data_2}')
                else:
                    changed_lines_count, recovered_lines_count, changed_regions_image = self.filtering._find_changed_lines(img_or_path, result_data_2)

                    if changed_lines_count or recovered_lines_count:
                        logger.opt(colors=True).info(f"<{engine_color}>{engine_instance_2.readable_name}</{engine_color}> found {changed_lines_count + recovered_lines_count} changed line(s) in {end_time - start_time:0.03f}s, re-OCRing with <{engine_color}>{engine_instance.readable_name}</{engine_color}>")

                        if output_format != 'json':
                            if changed_regions_image:
                                img_or_path = changed_regions_image

                        self.second_pass_thread.start()
                        self.second_pass_thread.submit_task(img_or_path, engine_instance, recovered_lines_count)

                second_pass_result = self.second_pass_thread.get_result()
                if second_pass_result:
                    engine_name, res, result_data, processing_time, recovered_lines_count = second_pass_result
                else:
                    return
            else:
                self.second_pass_thread.stop()

        if auto_pause_handler and auto_pause:
            auto_pause_handler.allow_auto_pause.clear()

        if not result_data:
            start_time = time.time()
            res, result_data = engine_instance(img_or_path)
            end_time = time.time()
            processing_time = end_time - start_time
            engine_name = engine_instance.readable_name
            recovered_lines_count = 0

        if not res:
            if auto_pause_handler and auto_pause:
                auto_pause_handler.stop_timer()
            logger.opt(colors=True).warning(f'<{engine_color}>{engine_name}</{engine_color}> reported an error after {processing_time:0.03f}s: {result_data}')
            return

        verbosity = config.get_general('verbosity')
        output_string = ''
        log_message = ''
        result_data_text = None

        if isinstance(result_data, OcrResult):
            unprocessed_text = self._extract_lines_from_result(result_data)

            if output_format == 'json':
                result_dict = asdict(result_data)
                output_string = json.dumps(result_dict, ensure_ascii=False)
                log_message = self._post_process(unprocessed_text, False)
            else:
                result_data_text = unprocessed_text
        else:
            result_data_text = result_data

        if result_data_text != None:
            if filter_text:
                text_to_process = self.filtering._find_changed_lines_text(result_data_text, result_data, two_pass_processing_active, recovered_lines_count)
                if self.screen_capture_periodic and not text_to_process:
                    if auto_pause_handler and auto_pause:
                        auto_pause_handler.allow_auto_pause.set()
                    return
                output_string = self._post_process(text_to_process, True)
            else:
                if self.filtering.furigana_filter and isinstance(result_data, OcrResult):
                    result_data_text = self.filtering._standalone_furigana_filter(result_data_text, result_data)
                output_string = self._post_process(result_data_text, False)
            log_message = output_string

        if verbosity != 0:
            if verbosity < -1:
                log_message_terminal = ': ' + log_message
            elif verbosity == -1:
                log_message_terminal = ''
            else:
                log_message_terminal = ': ' + (log_message if len(log_message) <= verbosity else log_message[:verbosity] + '[...]')

            logger.opt(colors=True).info(f'Text recognized in {processing_time:0.03f}s using <{engine_color}>{engine_name}</{engine_color}>{log_message_terminal}')

        if notify and config.get_general('notifications'):
            notifier.send(title='owocr', message='Text recognized: ' + log_message, urgency=get_notification_urgency())

        write_to = config.get_general('write_to')
        if write_to == 'websocket':
            websocket_server_thread.send_text(output_string)
        elif write_to == 'clipboard':
            pyperclipfix.copy(output_string)
        else:
            with Path(write_to).open('a', encoding='utf-8') as f:
                f.write(output_string + '\n')

        if auto_pause_handler and auto_pause:
            if not paused.is_set():
                auto_pause_handler.start_timer()
            else:
                auto_pause_handler.stop_timer()


def get_notification_urgency():
    if sys.platform == 'win32':
        return Urgency.Low
    return Urgency.Normal


def pause_handler(is_combo=True):
    global paused
    message = 'Unpaused!' if paused.is_set() else 'Paused!'

    if auto_pause_handler:
        auto_pause_handler.stop_timer()
    if is_combo:
        notifier.send(title='owocr', message=message, urgency=get_notification_urgency())
    logger.info(message)
    paused.clear() if paused.is_set() else paused.set()


def engine_change_handler(user_input='s', is_combo=True):
    global engine_index
    old_engine_index = engine_index

    if user_input.lower() == 's':
        if engine_index == len(engine_keys) - 1:
            engine_index = 0
        else:
            engine_index += 1
    elif user_input.lower() != '' and user_input.lower() in engine_keys:
        engine_index = engine_keys.index(user_input.lower())
    if engine_index != old_engine_index:
        new_engine_name = engine_instances[engine_index].readable_name
        if is_combo:
            notifier.send(title='owocr', message=f'Switched to {new_engine_name}', urgency=get_notification_urgency())
        engine_color = config.get_general('engine_color')
        logger.opt(colors=True).info(f'Switched to <{engine_color}>{new_engine_name}</{engine_color}>!')


def user_input_thread_run():
    def _terminate_handler():
        global terminated
        logger.info('Terminated!')
        terminated.set()

    if sys.platform == 'win32':
        import msvcrt
        while not terminated.is_set():
            if coordinate_selector_event.is_set():
                while coordinate_selector_event.is_set():
                    time.sleep(0.1)
            if msvcrt.kbhit():
                try:
                    user_input_bytes = msvcrt.getch()
                    user_input = user_input_bytes.decode()
                    if user_input.lower() in 'tq':
                        _terminate_handler()
                    elif user_input.lower() == 'p':
                        pause_handler(False)
                    else:
                        engine_change_handler(user_input, False)
                except UnicodeDecodeError:
                    pass
            else:
                time.sleep(0.2)
    else:
        import termios, select
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        new_settings = termios.tcgetattr(fd)
        new_settings[0] &= ~termios.IXON
        new_settings[3] &= ~(termios.ICANON | termios.ECHO)
        new_settings[6][termios.VMIN] = 1
        new_settings[6][termios.VTIME] = 0
        try:
            termios.tcsetattr(fd, termios.TCSANOW, new_settings)
            while not terminated.is_set():
                if coordinate_selector_event.is_set():
                    while coordinate_selector_event.is_set():
                        time.sleep(0.1)
                    termios.tcsetattr(fd, termios.TCSANOW, new_settings)
                rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
                if rlist:
                    user_input = sys.stdin.read(1)
                    if user_input.lower() in 'tq':
                        _terminate_handler()
                    elif user_input.lower() == 'p':
                        pause_handler(False)
                    else:
                        engine_change_handler(user_input, False)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def signal_handler(sig, frame):
    global terminated
    logger.info('Terminated!')
    terminated.set()


def on_window_closed(alive):
    global terminated
    if not (alive or terminated):
        logger.info('Window closed or error occurred, terminated!')
        terminated.set()


def on_screenshot_combo():
    screenshot_request_queue.put(True)


def on_coordinate_selector_combo():
    coordinate_selector_event.set()


def run():
    logger_level = 'DEBUG' if config.get_general('uwu') else 'INFO'
    logger.configure(handlers=[{'sink': sys.stderr, 'format': config.get_general('logger_format'), 'level': logger_level}])

    if config.has_config:
        logger.info('Parsed config file')
    else:
        logger.warning('No config file, defaults will be used.')
        if config.downloaded_config:
            logger.info(f'A default config file has been downloaded to {config.config_path}')

    global engine_instances
    global engine_keys
    output_format = config.get_general('output_format')
    engine_instances = []
    config_engines = []
    engine_keys = []
    default_engine = ''
    engine_secondary = ''

    if len(config.get_general('engines')) > 0:
        for config_engine in config.get_general('engines').split(','):
            config_engines.append(config_engine.strip().lower())

    for _,engine_class in sorted(inspect.getmembers(sys.modules[__name__], lambda x: hasattr(x, '__module__') and x.__module__ and __package__ + '.ocr' in x.__module__ and inspect.isclass(x) and hasattr(x, 'name'))):
        if len(config_engines) == 0 or engine_class.name in config_engines:

            if output_format == 'json' and not engine_class.coordinate_support:
                logger.warning(f"Skipping {engine_class.readable_name} as it does not support JSON output.")
                continue

            if config.get_engine(engine_class.name) == None:
                if engine_class.manual_language:
                    engine_instance = engine_class(language=config.get_general('language'))
                else:
                    engine_instance = engine_class()
            else:
                if engine_class.manual_language:
                    engine_instance = engine_class(config=config.get_engine(engine_class.name), language=config.get_general('language'))
                else:
                    engine_instance = engine_class(config=config.get_engine(engine_class.name))

            if engine_instance.available:
                engine_instances.append(engine_instance)
                engine_keys.append(engine_class.key)
                if config.get_general('engine') == engine_class.name:
                    default_engine = engine_class.key
                if config.get_general('engine_secondary') == engine_class.name and engine_class.local and engine_class.coordinate_support:
                    engine_secondary = engine_class.key

    if len(engine_keys) == 0:
        logger.error('No engines available!')
        sys.exit(1)

    global engine_index
    global engine_index_2
    global terminated
    global paused
    global notifier
    global auto_pause_handler
    global websocket_server_thread
    global screenshot_thread
    global image_queue
    global coordinate_selector_event
    non_path_inputs = ('screencapture', 'clipboard', 'websocket', 'unixsocket')
    read_from = config.get_general('read_from')
    read_from_secondary = config.get_general('read_from_secondary')
    read_from_path = None
    read_from_readable = []
    write_to = config.get_general('write_to')
    terminated = threading.Event()
    paused = threading.Event()
    if config.get_general('pause_at_startup'):
        paused.set()
    auto_pause = config.get_general('auto_pause')
    clipboard_thread = None
    websocket_server_thread = None
    screenshot_thread = None
    directory_watcher_thread = None
    unix_socket_server = None
    key_combo_listener = None
    auto_pause_handler = None
    engine_index = engine_keys.index(default_engine) if default_engine != '' else 0
    engine_index_2 = engine_keys.index(engine_secondary) if engine_secondary != '' else -1
    engine_color = config.get_general('engine_color')
    combo_pause = config.get_general('combo_pause')
    combo_engine_switch = config.get_general('combo_engine_switch')
    screen_capture_periodic = False
    screen_capture_on_combo = False
    coordinate_selector_event = threading.Event()
    notifier = DesktopNotifierSync()
    image_queue = queue.Queue()
    key_combos = {}

    if combo_pause != '':
        key_combos[combo_pause] = pause_handler
    if combo_engine_switch != '':
        key_combos[combo_engine_switch] = engine_change_handler

    if 'websocket' in (read_from, read_from_secondary) or write_to == 'websocket':
        websocket_server_thread = WebsocketServerThread('websocket' in (read_from, read_from_secondary))
        websocket_server_thread.start()
    if 'screencapture' in (read_from, read_from_secondary):
        global screenshot_request_queue
        screen_capture_delay_secs = config.get_general('screen_capture_delay_secs')
        screen_capture_combo = config.get_general('screen_capture_combo')
        coordinate_selector_combo = config.get_general('coordinate_selector_combo')
        last_screenshot_time = 0
        if screen_capture_combo != '':
            screen_capture_on_combo = True
            key_combos[screen_capture_combo] = on_screenshot_combo
        if coordinate_selector_combo != '':
            key_combos[coordinate_selector_combo] = on_coordinate_selector_combo
        if screen_capture_delay_secs != -1:
            global periodic_screenshot_queue
            periodic_screenshot_queue = queue.Queue()
            screen_capture_periodic = True
        if not (screen_capture_on_combo or screen_capture_periodic):
            logger.error('screen_capture_delay_secs or screen_capture_combo need to be valid values')
            sys.exit(1)
        screenshot_request_queue = queue.Queue()
        screenshot_thread = ScreenshotThread()
        screenshot_thread.start()
        read_from_readable.append('screen capture')
    if 'websocket' in (read_from, read_from_secondary):
        read_from_readable.append('websocket')
    if 'unixsocket' in (read_from, read_from_secondary):
        if sys.platform == 'win32':
            logger.error('"unixsocket" is not currently supported on Windows')
            sys.exit(1)
        socket_path = Path('/tmp/owocr.sock')
        if socket_path.exists():
            socket_path.unlink()
        unix_socket_server = socketserver.ThreadingUnixStreamServer(str(socket_path), RequestHandler)
        unix_socket_server_thread = threading.Thread(target=unix_socket_server.serve_forever, daemon=True)
        unix_socket_server_thread.start()
        read_from_readable.append('unix socket')
    if 'clipboard' in (read_from, read_from_secondary):
        clipboard_thread = ClipboardThread()
        clipboard_thread.start()
        read_from_readable.append('clipboard')
    if any(i and i not in non_path_inputs for i in (read_from, read_from_secondary)):
        if all(i and i not in non_path_inputs for i in (read_from, read_from_secondary)):
            logger.error("read_from and read_from_secondary can't both be directory paths")
            sys.exit(1)
        delete_images = config.get_general('delete_images')
        read_from_path = Path(read_from) if read_from not in non_path_inputs else Path(read_from_secondary)
        if not read_from_path.is_dir():
            logger.error('read_from and read_from_secondary must be either "websocket", "unixsocket", "clipboard", "screencapture", or a path to a directory')
            sys.exit(1)
        directory_watcher_thread = DirectoryWatcher(read_from_path)
        directory_watcher_thread.start()
        read_from_readable.append(f'directory {read_from_path}')

    output_result = OutputResult()

    if len(key_combos) > 0:
        key_combo_listener = keyboard.GlobalHotKeys(key_combos)
        key_combo_listener.start()

    if write_to in ('clipboard', 'websocket'):
        write_to_readable = write_to
    else:
        if Path(write_to).suffix.lower() != '.txt':
            logger.error('write_to must be either "websocket", "clipboard" or a path to a text file')
            sys.exit(1)
        write_to_readable = f'file {write_to}'

    process_queue = (any(i in ('clipboard', 'websocket', 'unixsocket') for i in (read_from, read_from_secondary)) or read_from_path or screen_capture_on_combo)
    signal.signal(signal.SIGINT, signal_handler)
    if auto_pause != 0:
        auto_pause_handler = AutopauseTimer()
    user_input_thread = threading.Thread(target=user_input_thread_run, daemon=True)
    user_input_thread.start()

    logger.opt(colors=True).info(f"Reading from {' and '.join(read_from_readable)}, writing to {write_to_readable} using <{engine_color}>{engine_instances[engine_index].readable_name}</{engine_color}>{' (paused)' if paused.is_set() else ''}")

    while not terminated.is_set():
        img = None
        skip_waiting = False
        filter_text = False
        auto_pause = True
        notify = False

        if process_queue:
            try:
                img, is_screen_capture = image_queue.get_nowait()
                if not screen_capture_periodic and is_screen_capture:
                    filter_text = True
                if is_screen_capture:
                    auto_pause = False
                notify = True
            except queue.Empty:
                pass

        if (not img) and screen_capture_periodic:
            if (not paused.is_set()) and screenshot_thread.screencapture_window_active and screenshot_thread.screencapture_window_visible and (time.time() - last_screenshot_time) > screen_capture_delay_secs:
                if periodic_screenshot_queue.empty() and screenshot_request_queue.empty():
                    screenshot_request_queue.put(False)
                try:
                    img = periodic_screenshot_queue.get(timeout=0.5)
                    filter_text = True
                    last_screenshot_time = time.time()
                except queue.Empty:
                    skip_waiting = True
                    pass

        if img == 0:
            on_window_closed(False)
            terminated.set()
            break
        elif img:
            output_result(img, filter_text, auto_pause, notify)
            if isinstance(img, Path) and delete_images:
                Path.unlink(img)

        if not img and not skip_waiting:
            time.sleep(0.1)

    terminate_selector_if_running()
    user_input_thread.join()
    auto_pause_handler.stop()
    output_result.second_pass_thread.stop()
    if websocket_server_thread:
        websocket_server_thread.stop_server()
        websocket_server_thread.join()
    if clipboard_thread:
        if sys.platform == 'win32':
            win32api.PostThreadMessage(clipboard_thread.thread_id, win32con.WM_QUIT, 0, 0)
        clipboard_thread.join()
    if directory_watcher_thread:
        directory_watcher_thread.join()
    if unix_socket_server:
        unix_socket_server.shutdown()
        unix_socket_server_thread.join()
    if screenshot_thread:
        screenshot_thread.join()
    if key_combo_listener:
        key_combo_listener.stop()
