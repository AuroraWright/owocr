from ...ocr.gsm_ocr_config import set_dpi_awareness

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
        CGImageGetWidth, CGImageGetHeight, CGDataProviderCopyData, CGImageGetDataProvider, CGImageGetBytesPerRow
    from ScreenCaptureKit import SCContentFilter, SCScreenshotManager, SCShareableContent, SCStreamConfiguration, SCCaptureResolutionBest
except ImportError:
    pass

import signal
import threading
from pathlib import Path
import queue
import io
import re
import logging
import inspect
import time

import pyperclipfix
import mss
import asyncio
import websockets
import socketserver
import queue

from datetime import datetime
from PIL import Image, ImageDraw, UnidentifiedImageError
from loguru import logger
from pynput import keyboard
from desktop_notifier import DesktopNotifierSync
import psutil

import inspect
from .ocr import *
try:
    from .secret import *
except ImportError:
    pass
from .config import Config
from .screen_coordinate_picker import get_screen_selection
from ...configuration import get_temporary_directory, get_app_directory

config = None


class ClipboardThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.ignore_flag = config.get_general('ignore_flag')
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
        if msg == WM_CLIPBOARDUPDATE and timestamp - self.last_update > 1 and not paused:
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
                    clipboard_text = ''
                    if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
                        clipboard_text = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
                    if self.ignore_flag or clipboard_text != '*ocr_ignore*':
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
                from AppKit import NSPasteboard, NSPasteboardTypeTIFF, NSPasteboardTypeString
                pasteboard = NSPasteboard.generalPasteboard()
                count = pasteboard.changeCount()
            else:
                from PIL import ImageGrab
            process_clipboard = False
            img = None

            while not terminated:
                if paused:
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
                                    clipboard_text = ''
                                    if NSPasteboardTypeString in pasteboard.types():
                                        clipboard_text = pasteboard.stringForType_(NSPasteboardTypeString)
                                    if self.ignore_flag or clipboard_text != '*ocr_ignore*':
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
                                    (self.ignore_flag or pyperclipfix.paste() != '*ocr_ignore*') and \
                                    (not self.are_images_identical(img, old_img))):
                                image_queue.put((img, False))

                    process_clipboard = True

                if not terminated:
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
                old_paths.add(get_path_key(path))

        while not terminated:
            if paused:
                sleep_time = 0.5
            else:
                sleep_time = self.delay_secs
                for path in self.path.iterdir():
                    if path.suffix.lower() in self.allowed_extensions:
                        path_key = self.get_path_key(path)
                        if path_key not in old_paths:
                            old_paths.add(path_key)

                            if not paused:
                                image_queue.put((path, False))

            if not terminated:
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
                if self.read and not paused:
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

        if not paused:
            image_queue.put((img, False))
            conn.sendall(b'True')
        else:
            conn.sendall(b'False')


class TextFiltering:
    accurate_filtering = False

    def __init__(self):
        from pysbd import Segmenter
        self.segmenter = Segmenter(language=lang, clean=True)
        self.kana_kanji_regex = re.compile(r'[\u3041-\u3096\u30A1-\u30FA\u4E00-\u9FFF]')
        self.chinese_common_regex = re.compile(r'[\u4E00-\u9FFF]')
        self.english_regex = re.compile(r'[a-zA-Z0-9.,!?;:"\'()\[\]{}]')
        self.kana_kanji_regex = re.compile(r'[\u3041-\u3096\u30A1-\u30FA\u4E00-\u9FFF]')
        self.chinese_common_regex = re.compile(r'[\u4E00-\u9FFF]')
        self.english_regex = re.compile(r'[a-zA-Z0-9.,!?;:"\'()\[\]{}]')
        self.korean_regex = re.compile(r'[\uAC00-\uD7AF]')
        self.arabic_regex = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
        self.russian_regex = re.compile(r'[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F\u1C80-\u1C8F]')
        self.greek_regex = re.compile(r'[\u0370-\u03FF\u1F00-\u1FFF]')
        self.hebrew_regex = re.compile(r'[\u0590-\u05FF\uFB1D-\uFB4F]')
        self.thai_regex = re.compile(r'[\u0E00-\u0E7F]')
        self.latin_extended_regex = re.compile(
            r'[a-zA-Z\u00C0-\u00FF\u0100-\u017F\u0180-\u024F\u0250-\u02AF\u1D00-\u1D7F\u1D80-\u1DBF\u1E00-\u1EFF\u2C60-\u2C7F\uA720-\uA7FF\uAB30-\uAB6F]')
        try:
            from transformers import pipeline, AutoTokenizer
            import torch
            logging.getLogger('transformers').setLevel(logging.ERROR)

            model_ckpt = 'papluca/xlm-roberta-base-language-detection'
            tokenizer = AutoTokenizer.from_pretrained(
                model_ckpt,
                use_fast = False
            )

            if torch.cuda.is_available():
                device = 0
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = -1
            self.pipe = pipeline('text-classification', model=model_ckpt, tokenizer=tokenizer, device=device)
            self.accurate_filtering = True
        except:
            import langid
            self.classify = langid.classify

    def __call__(self, text, last_result):
        orig_text = self.segmenter.segment(text)

        orig_text_filtered = []
        for block in orig_text:
            if lang == "ja":
                block_filtered = self.kana_kanji_regex.findall(block)
            elif lang == "zh":
                block_filtered = self.chinese_common_regex.findall(block)
            elif lang == "ko":
                block_filtered = self.korean_regex.findall(block)
            elif lang == "ar":
                block_filtered = self.arabic_regex.findall(block)
            elif lang == "ru":
                block_filtered = self.russian_regex.findall(block)
            elif lang == "el":
                block_filtered = self.greek_regex.findall(block)
            elif lang == "he":
                block_filtered = self.hebrew_regex.findall(block)
            elif lang == "th":
                block_filtered = self.thai_regex.findall(block)
            elif lang in ["en", "fr", "de", "es", "it", "pt", "nl", "sv", "da", "no",
                          "fi"]:  # Many European languages use extended Latin
                block_filtered = self.latin_extended_regex.findall(block)
            else:
                block_filtered = self.english_regex.findall(block)

            if block_filtered:
                orig_text_filtered.append(''.join(block_filtered))
            else:
                orig_text_filtered.append(None)

        if last_result and last_result[1] == engine_index:
            last_text = last_result[0]
        else:
            last_text = []

        new_blocks = []
        for idx, block in enumerate(orig_text):
            if orig_text_filtered[idx] and (orig_text_filtered[idx] not in last_text):
                new_blocks.append(block)

        final_blocks = []
        if self.accurate_filtering:
            detection_results = self.pipe(new_blocks, top_k=3, truncation=True)
            for idx, block in enumerate(new_blocks):
                for result in detection_results[idx]:
                    if result['label'] == lang:
                        final_blocks.append(block)
                        break
        else:
            for block in new_blocks:
                print(block)
                if lang not in ["ja", "zh"] or self.classify(block)[0] == lang:
                    final_blocks.append(block)

        text = '\n'.join(final_blocks)
        return text, orig_text_filtered


class ScreenshotClass:
    def __init__(self, screen_capture_area, screen_capture_window, screen_capture_exclusions, screen_capture_only_active_windows, screen_capture_areas):
        self.macos_window_tracker_instance = None
        self.windows_window_tracker_instance = None
        self.screencapture_window_active = True
        self.screencapture_window_visible = True
        self.custom_left = None
        self.screen_capture_exclusions = screen_capture_exclusions
        self.screen_capture_window = screen_capture_window
        self.areas = []
        if screen_capture_area == '':
            self.screencapture_mode = 0
        elif screen_capture_area.startswith('screen_'):
            parts = screen_capture_area.split('_')
            if len(parts) != 2 or not parts[1].isdigit():
                raise ValueError('Invalid screen_capture_area')
            screen_capture_monitor = int(parts[1])
            self.screencapture_mode = 1
        elif len(screen_capture_area.split(',')) == 4:
            self.screencapture_mode = 3
        else:
            self.screencapture_mode = 2
            self.screen_capture_window = screen_capture_area
        if self.screen_capture_window:
            self.screencapture_mode = 2

        if self.screencapture_mode != 2:
            self.sct = mss.mss()

            if self.screencapture_mode == 1:
                mon = self.sct.monitors
                if len(mon) <= screen_capture_monitor:
                    raise ValueError('Invalid monitor number in screen_capture_area')
                coord_left = mon[screen_capture_monitor]['left']
                coord_top = mon[screen_capture_monitor]['top']
                coord_width = mon[screen_capture_monitor]['width']
                coord_height = mon[screen_capture_monitor]['height']
            elif self.screencapture_mode == 3:
                coord_left, coord_top, coord_width, coord_height = [int(c.strip()) for c in screen_capture_area.split(',')]
            else:
                logger.opt(ansi=True).info('Launching screen coordinate picker')
                screen_selection = get_screen_selection()
                if not screen_selection:
                    raise ValueError('Picker window was closed or an error occurred')
                screen_capture_monitor = screen_selection['monitor']
                x, y, coord_width, coord_height = screen_selection['coordinates']
                if coord_width > 0 and coord_height > 0:
                    coord_top = screen_capture_monitor['top'] + y
                    coord_left = screen_capture_monitor['left'] + x
                else:
                    logger.opt(ansi=True).info('Selection is empty, selecting whole screen')
                    coord_left = screen_capture_monitor['left']
                    coord_top = screen_capture_monitor['top']
                    coord_width = screen_capture_monitor['width']
                    coord_height = screen_capture_monitor['height']

            self.sct_params = {'top': coord_top, 'left': coord_left, 'width': coord_width, 'height': coord_height}
            logger.opt(ansi=True).info(f'Selected coordinates: {coord_left},{coord_top},{coord_width},{coord_height}')
        if screen_capture_areas:
            for area in screen_capture_areas:
                if len(area.split(',')) == 4:
                    self.areas.append(([int(c.strip()) for c in area.split(',')]))
        else:
            if len(screen_capture_area.split(',')) == 4:
                self.areas.append(([int(c.strip()) for c in screen_capture_area.split(',')]))

        self.areas.sort(key=lambda rect: (rect[1], rect[0]))


        if self.screencapture_mode == 2 or self.screen_capture_window:
            self.screen_capture_only_active_windows = screen_capture_only_active_windows
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

                if screen_capture_window in window_titles:
                    window_index = window_titles.index(screen_capture_window)
                else:
                    for t in window_titles:
                        if screen_capture_window in t:
                            window_index = window_titles.index(t)
                            break

                if not window_index:
                    raise ValueError(area_invalid_error)

                self.window_id = window_ids[window_index]
                window_title = window_titles[window_index]

                if self.screen_capture_only_active_windows:
                    self.macos_window_tracker_instance = threading.Thread(target=self.macos_window_tracker)
                    self.macos_window_tracker_instance.start()
                logger.opt(ansi=True).info(f'Selected window: {window_title}')
            elif sys.platform == 'win32':
                self.window_handle, window_title = self.get_windows_window_handle(screen_capture_window)

                if not self.window_handle:
                    raise ValueError(area_invalid_error)

                set_dpi_awareness()

                self.windows_window_tracker_instance = threading.Thread(target=self.windows_window_tracker)
                self.windows_window_tracker_instance.start()
                logger.opt(ansi=True).info(f'Selected window: {window_title}')
            else:
                raise ValueError('Window capture is only currently supported on Windows and macOS')

    def __del__(self):
        if self.macos_window_tracker_instance:
            self.macos_window_tracker_instance.join()
        elif self.windows_window_tracker_instance:
            self.windows_window_tracker_instance.join()

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
        while not terminated:
            found = win32gui.IsWindow(self.window_handle)
            if not found:
                break
            if self.screen_capture_only_active_windows:
                self.screencapture_window_active = self.window_handle == win32gui.GetForegroundWindow()
            else:
                self.screencapture_window_visible = not win32gui.IsIconic(self.window_handle)
            time.sleep(0.2)
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
                scale = content_filter.pointPixelScale()
                width = frame.size.width * scale
                height = frame.size.height * scale
                configuration = SCStreamConfiguration.alloc().init()
                configuration.setSourceRect_(CGRectMake(0, 0, frame.size.width, frame.size.height))
                configuration.setWidth_(width)
                configuration.setHeight_(height)
                configuration.setShowsCursor_(False)
                configuration.setCaptureResolution_(SCCaptureResolutionBest)
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
        while found and not terminated:
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
            time.sleep(0.2)
        if not found:
            on_window_closed(False)

    def __call__(self):
        if self.screencapture_mode == 2 or self.screen_capture_window:
            if sys.platform == 'darwin':
                with objc.autorelease_pool():
                    if self.old_macos_screenshot_api:
                        cg_image = CGWindowListCreateImageFromArray(CGRectNull, [self.window_id], kCGWindowImageBoundsIgnoreFraming)
                    else:
                        self.capture_macos_window_screenshot(self.window_id)
                        try:
                            cg_image = self.screencapturekit_queue.get(timeout=0.5)
                        except queue.Empty:
                            cg_image = None
                    if not cg_image:
                        return 0
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
                    return 0
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
        else:
            sct_img = self.sct.grab(self.sct_params)
            img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')

        import random  # Ensure this is imported at the top of the file if not already
        rand_int = random.randint(1, 20)  # Executes only once out of 10 times

        if rand_int == 1:  # Executes only once out of 10 times
            img.save(os.path.join(get_temporary_directory(), 'before_crop.png'), 'PNG')

        if self.screen_capture_exclusions:
            img = img.convert("RGBA")
            draw = ImageDraw.Draw(img)
            for exclusion in self.screen_capture_exclusions:
                left, top, width, height = exclusion
                draw.rectangle((left, top, left + width, top + height), fill=(0, 0, 0, 0))

        cropped_sections = []
        start = time.time()
        for area in self.areas:
            cropped_sections.append(img.crop((area[0], area[1], area[0] + area[2], area[1] + area[3])))

        # if len(cropped_sections) > 1:
        #     combined_width = sum(section.width for section in cropped_sections)
        #     combined_height = max(section.height for section in cropped_sections)
        #     combined_img = Image.new("RGBA", (combined_width, combined_height))
        #
        #     x_offset = 0
        #     for section in cropped_sections:
        #         combined_img.paste(section, (x_offset, 0))
        #         x_offset += section.width
        #
        #     img = combined_img
        if len(cropped_sections) > 1:
            combined_width = max(section.width for section in cropped_sections)
            combined_height = sum(section.height for section in cropped_sections) + (len(cropped_sections) - 1) * 10  # Add space for gaps
            combined_img = Image.new("RGBA", (combined_width, combined_height))

            y_offset = 0
            for section in cropped_sections:
                combined_img.paste(section, (0, y_offset))
                y_offset += section.height + 50  # Add gap between sections

            img = combined_img
        elif cropped_sections:
            img = cropped_sections[0]

        if rand_int == 1:
            img.save(os.path.join(get_temporary_directory(), 'after_crop.png'), 'PNG')
            print(f'OCR images saved to {get_temporary_directory()} if debugging is needed, this is 1/20 chance')

        return img


class AutopauseTimer:
    def __init__(self, timeout):
        self.stop_event = threading.Event()
        self.timeout = timeout
        self.timer_thread = None

    def __del__(self):
        self.stop()

    def start(self):
        self.stop()
        self.stop_event.clear()
        self.timer_thread = threading.Thread(target=self._countdown)
        self.timer_thread.start()

    def stop(self):
        if not self.stop_event.is_set() and self.timer_thread and self.timer_thread.is_alive():
            self.stop_event.set()
            self.timer_thread.join()

    def _countdown(self):
        seconds = self.timeout
        while seconds > 0 and not self.stop_event.is_set() and not terminated:
            time.sleep(1)
            seconds -= 1
        if not self.stop_event.is_set():
            self.stop_event.set()
            if not (paused or terminated):
                pause_handler(True)


def pause_handler(is_combo=True):
    global paused
    message = 'Unpaused!' if paused else 'Paused!'

    if auto_pause_handler:
        auto_pause_handler.stop()
    if is_combo:
        notifier.send(title='owocr', message=message)
    logger.info(message)
    paused = not paused


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
            notifier.send(title='owocr', message=f'Switched to {new_engine_name}')
        engine_color = config.get_general('engine_color')
        logger.opt(ansi=True).info(f'Switched to <{engine_color}>{new_engine_name}</{engine_color}>!')


def user_input_thread_run():
    def _terminate_handler():
        global terminated
        logger.info('Terminated!')
        terminated = True

    if sys.platform == 'win32':
        import msvcrt
        while not terminated:
            user_input_bytes = msvcrt.getch()
            try:
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
        import tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not terminated:
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
    terminated = True


def on_window_closed(alive):
    global terminated
    if not (alive or terminated):
        logger.info('Window closed or error occurred, terminated!')
        terminated = True


def on_screenshot_combo():
    if not paused:
        img = take_screenshot()
        image_queue.put((img, True))


def on_window_minimized(minimized):
    global screencapture_window_visible
    screencapture_window_visible = not minimized


def process_and_write_results(img_or_path, write_to=None, last_result=None, filtering=None, notify=None, engine=None, ocr_start_time=None):
    global engine_index
    if auto_pause_handler:
        auto_pause_handler.stop()
    if engine:
        for i, instance in enumerate(engine_instances):
            if instance.name.lower() in engine.lower():
                engine_instance = instance
                if last_result:
                    last_result = (last_result[0], i)
                break
    else:
        engine_instance = engine_instances[engine_index]

    start_time = time.time()
    res, text = engine_instance(img_or_path)
    end_time = time.time()

    orig_text = []
    engine_color = config.get_general('engine_color')
    # print(filtering)
    #
    #
    # print(lang)

    # print(last_result)
    # print(engine_index)

    if res:
        if filtering:
            text, orig_text = filtering(text, last_result)
        if lang == "ja" or lang == "zh":
            text = post_process(text)
        logger.opt(ansi=True).info(f'Text recognized in {end_time - start_time:0.03f}s using <{engine_color}>{engine_instance.readable_name}</{engine_color}>: {text}')
        if notify and config.get_general('notifications'):
            notifier.send(title='owocr', message='Text recognized: ' + text)

        if write_to == 'websocket':
            websocket_server_thread.send_text(text)
        elif write_to == 'clipboard':
            pyperclipfix.copy(text)
        elif write_to == "callback":
            txt_callback(text, orig_text, ocr_start_time, img_or_path, bool(engine), filtering)
        elif write_to:
            with Path(write_to).open('a', encoding='utf-8') as f:
                f.write(text + '\n')

        if auto_pause_handler and not paused:
            auto_pause_handler.start()
    else:
        logger.opt(ansi=True).info(f'<{engine_color}>{engine_instance.readable_name}</{engine_color}> reported an error after {end_time - start_time:0.03f}s: {text}')

    # print(orig_text)
    # print(text)

    return orig_text, text


def get_path_key(path):
    return path, path.lstat().st_mtime


def init_config(parse_args=True):
    global config
    config = Config(parse_args)


def run(read_from=None,
        read_from_secondary=None,
        write_to=None,
        engine=None,
        pause_at_startup=None,
        ignore_flag=None,
        delete_images=None,
        notifications=None,
        auto_pause=0,
        combo_pause=None,
        combo_engine_switch=None,
        screen_capture_area=None,
        screen_capture_areas=None,
        screen_capture_exclusions=None,
        screen_capture_window=None,
        screen_capture_delay_secs=None,
        screen_capture_only_active_windows=None,
        screen_capture_combo=None,
        stop_running_flag=None,
        screen_capture_event_bus=None,
        text_callback=None,
        language=None,
        monitor_index=None,
        ocr2=None,
        gsm_ocr_config=None,
        ):
    """
    Japanese OCR client

    Runs OCR in the background.
    It can read images copied to the system clipboard or placed in a directory, images sent via a websocket or a Unix domain socket, or directly capture a screen (or a portion of it) or a window.
    Recognized texts can be either saved to system clipboard, appended to a text file or sent via a websocket.

    :param read_from: Specifies where to read input images from. Can be either "clipboard", "websocket", "unixsocket" (on macOS/Linux), "screencapture", or a path to a directory.
    :param write_to: Specifies where to save recognized texts to. Can be either "clipboard", "websocket", or a path to a text file.
    :param delay_secs: How often to check for new images, in seconds.
    :param engine: OCR engine to use. Available: "mangaocr", "glens", "glensweb", "bing", "gvision", "avision", "alivetext", "azure", "winrtocr", "oneocr", "easyocr", "rapidocr", "ocrspace".
    :param pause_at_startup: Pause at startup.
    :param ignore_flag: Process flagged clipboard images (images that are copied to the clipboard with the *ocr_ignore* string).
    :param delete_images: Delete image files after processing when reading from a directory.
    :param notifications: Show an operating system notification with the detected text.
    :param auto_pause: Automatically pause the program after the specified amount of seconds since the last successful text recognition. Will be ignored when reading with screen capture. 0 to disable.
    :param combo_pause: Specifies a combo to wait on for pausing the program. As an example: "<ctrl>+<shift>+p". The list of keys can be found here: https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key
    :param combo_engine_switch: Specifies a combo to wait on for switching the OCR engine. As an example: "<ctrl>+<shift>+a". To be used with combo_pause. The list of keys can be found here: https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key
    :param screen_capture_area: Specifies area to target when reading with screen capture. Can be either empty (automatic selector), a set of coordinates (x,y,width,height), "screen_N" (captures a whole screen, where N is the screen number starting from 1) or a window name (the first matching window title will be used).
    :param screen_capture_delay_secs: Specifies the delay (in seconds) between screenshots when reading with screen capture.
    :param screen_capture_only_active_windows: When reading with screen capture and screen_capture_area is a window name, specifies whether to only target the window while it's active.
    :param screen_capture_combo: When reading with screen capture, specifies a combo to wait on for taking a screenshot instead of using the delay. As an example: "<ctrl>+<shift>+s". The list of keys can be found here: https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key
    """

    if read_from is None:
        read_from = config.get_general('read_from')

    if read_from_secondary is None:
        read_from_secondary = config.get_general('read_from_secondary')

    if screen_capture_area is None:
        screen_capture_area = config.get_general('screen_capture_area')

    if screen_capture_only_active_windows is None:
        screen_capture_only_active_windows = config.get_general('screen_capture_only_active_windows')

    if screen_capture_exclusions is None:
        screen_capture_exclusions = config.get_general('screen_capture_exclusions')

    if screen_capture_window is None:
        screen_capture_window = config.get_general('screen_capture_window')

    if screen_capture_delay_secs is None:
        screen_capture_delay_secs = config.get_general('screen_capture_delay_secs')

    if screen_capture_combo is None:
        screen_capture_combo = config.get_general('screen_capture_combo')

    if stop_running_flag is None:
        stop_running_flag = config.get_general('stop_running_flag')

    if screen_capture_event_bus is None:
        screen_capture_event_bus = config.get_general('screen_capture_event_bus')

    if text_callback is None:
        text_callback = config.get_general('text_callback')

    if write_to is None:
        write_to = config.get_general('write_to')

    if language is None:
        language = config.get_general('language', "ja")

    logger.configure(handlers=[{'sink': sys.stderr, 'format': config.get_general('logger_format')}])

    if config.has_config:
        logger.info('Parsed config file')
    else:
        logger.warning('No config file, defaults will be used.')
        if config.downloaded_config:
            logger.info(f'A default config file has been downloaded to {config.config_path}')

    global engine_instances
    global engine_keys
    global lang
    lang = language
    engine_instances = []
    config_engines = []
    engine_keys = []
    default_engine = ''

    if len(config.get_general('engines')) > 0:
        for config_engine in config.get_general('engines').split(','):
            config_engines.append(config_engine.strip().lower())

    for _, engine_class in sorted(inspect.getmembers(sys.modules[__name__],
                                                     lambda x: hasattr(x, '__module__') and x.__module__ and (
                                                             __package__ + '.ocr' in x.__module__ or __package__ + '.secret' in x.__module__) and inspect.isclass(
                                                         x))):
        if len(config_engines) == 0 or engine_class.name in config_engines:
            if config.get_engine(engine_class.name) == None:
                engine_instance = engine_class()
            else:
                engine_instance = engine_class(config.get_engine(engine_class.name))

            if engine_instance.available:
                engine_instances.append(engine_instance)
                engine_keys.append(engine_class.key)
                if engine == engine_class.name:
                    default_engine = engine_class.key

    if len(engine_keys) == 0:
        msg = 'No engines available!'
        raise NotImplementedError(msg)

    global engine_index
    global terminated
    global paused
    global just_unpaused
    global first_pressed
    global auto_pause_handler
    global notifier
    global websocket_server_thread
    global image_queue
    custom_left = None
    terminated = False
    paused = pause_at_startup
    just_unpaused = True
    first_pressed = None
    auto_pause_handler = None
    engine_index = engine_keys.index(default_engine) if default_engine != '' else 0
    engine_color = config.get_general('engine_color')
    prefix_to_use = ""
    delay_secs = config.get_general('delay_secs')

    non_path_inputs = ('screencapture', 'clipboard', 'websocket', 'unixsocket')
    read_from_path = None
    read_from_readable = []
    terminated = False
    paused = config.get_general('pause_at_startup')
    auto_pause = config.get_general('auto_pause')
    clipboard_thread = None
    websocket_server_thread = None
    directory_watcher_thread = None
    unix_socket_server = None
    key_combo_listener = None
    filtering = None
    auto_pause_handler = None
    engine_index = engine_keys.index(default_engine) if default_engine != '' else 0
    engine_color = config.get_general('engine_color')
    combo_pause = config.get_general('combo_pause')
    combo_engine_switch = config.get_general('combo_engine_switch')
    screen_capture_on_combo = False
    notifier = DesktopNotifierSync()
    image_queue = queue.Queue()
    key_combos = {}

    if combo_pause != '':
        key_combos[combo_pause] = pause_handler
    if combo_engine_switch:
        if combo_pause:
            key_combos[combo_engine_switch] = engine_change_handler
        else:
            raise ValueError('combo_pause must also be specified')

    if 'websocket' in (read_from, read_from_secondary) or write_to == 'websocket':
        websocket_server_thread = WebsocketServerThread('websocket' in (read_from, read_from_secondary))
        websocket_server_thread.start()

    if write_to == "callback" and text_callback:
        global txt_callback
        txt_callback = text_callback

    if 'screencapture' in (read_from, read_from_secondary):
        global take_screenshot
        screen_capture_combo = config.get_general('screen_capture_combo')
        last_screenshot_time = 0
        last_result = ([], engine_index)
        if screen_capture_combo != '':
            screen_capture_on_combo = True
            key_combos[screen_capture_combo] = on_screenshot_combo
        take_screenshot = ScreenshotClass(screen_capture_area, screen_capture_window, screen_capture_exclusions, screen_capture_only_active_windows, screen_capture_areas)
        # global_take_screenshot = ScreenshotClass(screen_capture_area, screen_capture_window, screen_capture_exclusions, screen_capture_only_active_windows, rectangle)
        filtering = TextFiltering()
        read_from_readable.append('screen capture')
    if 'websocket' in (read_from, read_from_secondary):
        read_from_readable.append('websocket')
    if 'unixsocket' in (read_from, read_from_secondary):
        if sys.platform == 'win32':
            raise ValueError('"unixsocket" is not currently supported on Windows')
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
            raise ValueError("read_from and read_from_secondary can't both be directory paths")
        delete_images = config.get_general('delete_images')
        read_from_path = Path(read_from) if read_from not in non_path_inputs else Path(read_from_secondary)
        if not read_from_path.is_dir():
            raise ValueError('read_from and read_from_secondary must be either "websocket", "unixsocket", "clipboard", "screencapture", or a path to a directory')
        directory_watcher_thread = DirectoryWatcher(read_from_path)
        directory_watcher_thread.start()
        read_from_readable.append(f'directory {read_from_path}')

    if len(key_combos) > 0:
        key_combo_listener = keyboard.GlobalHotKeys(key_combos)
        key_combo_listener.start()

    if write_to in ('clipboard', 'websocket', 'callback'):
        write_to_readable = write_to
    else:
        if Path(write_to).suffix.lower() != '.txt':
            raise ValueError('write_to must be either "websocket", "clipboard" or a path to a text file')
        write_to_readable = f'file {write_to}'

    process_queue = (any(i in ('clipboard', 'websocket', 'unixsocket') for i in (read_from, read_from_secondary)) or read_from_path or screen_capture_on_combo)
    process_screenshots = 'screencapture' in (read_from, read_from_secondary) and not screen_capture_on_combo
    if threading.current_thread() == threading.main_thread():
        signal.signal(signal.SIGINT, signal_handler)
    if (not process_screenshots) and auto_pause != 0:
        auto_pause_handler = AutopauseTimer(auto_pause)
    user_input_thread = threading.Thread(target=user_input_thread_run, daemon=True)
    user_input_thread.start()
    logger.opt(ansi=True).info(f"Reading from {' and '.join(read_from_readable)}, writing to {write_to_readable} using <{engine_color}>{engine_instances[engine_index].readable_name}</{engine_color}>{' (paused)' if paused else ''}")

    while not terminated:
        ocr_start_time = datetime.now()
        start_time = time.time()
        img = None
        filter_img = False

        if process_queue:
            try:
                img, filter_img = image_queue.get(timeout=0.1)
                notify = True
            except queue.Empty:
                pass

        if (not img) and process_screenshots:
            if (not paused) and take_screenshot.screencapture_window_active and take_screenshot.screencapture_window_visible and (time.time() - last_screenshot_time) > screen_capture_delay_secs:
                img = take_screenshot()
                filter_img = True
                notify = False
                last_screenshot_time = time.time()

        if img == 0:
            on_window_closed(False)
            terminated = True
            break
        elif img:
            if filter_img:
                res, _ = process_and_write_results(img, write_to, last_result, filtering, notify, ocr_start_time=ocr_start_time)
                if res:
                    last_result = (res, engine_index)
            else:
                process_and_write_results(img, write_to, None, notify=notify, ocr_start_time=ocr_start_time, engine=ocr2)
            if isinstance(img, Path):
                if delete_images:
                    Path.unlink(img)

        elapsed_time = time.time() - start_time
        if (not terminated) and elapsed_time < 0.1:
            time.sleep(0.1 - elapsed_time)

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
    if key_combo_listener:
        key_combo_listener.stop()
