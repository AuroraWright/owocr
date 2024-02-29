import sys
import signal
import time
import threading
from pathlib import Path

import fire
import numpy as np
import pyperclipfix
import mss
import asyncio
import websockets
import socketserver
import queue
import io

from PIL import Image
from PIL import UnidentifiedImageError
from loguru import logger
from pynput import keyboard
from desktop_notifier import DesktopNotifier

import inspect
from owocr.ocr import *
from owocr.config import Config

try:
    import win32gui
    import win32api 
    import win32con
    import win32clipboard
    import pywintypes
    import ctypes
except ImportError:
    pass

try:
    import pywinctl
except ImportError:
    pass

try:
    import objc
    from AppKit import NSData, NSImage, NSBitmapImageRep, NSDeviceRGBColorSpace, NSGraphicsContext, NSZeroPoint, NSZeroRect, NSCompositingOperationCopy
    from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionAll, kCGWindowListOptionOnScreenOnly, kCGWindowListExcludeDesktopElements, kCGWindowName, kCGNullWindowID
    import psutil
except ImportError:
    pass


config = None


class WindowsClipboardThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.last_update = time.time()

    def process_message(self, hwnd: int, msg: int, wparam: int, lparam: int):
        WM_CLIPBOARDUPDATE = 0x031D
        timestamp = time.time()
        if msg == WM_CLIPBOARDUPDATE and timestamp - self.last_update > 1 and not paused:
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_BITMAP):
                clipboard_event.set()
                self.last_update = timestamp
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
        hwnd = self.create_window()
        self.thread_id = win32api.GetCurrentThreadId()
        ctypes.windll.user32.AddClipboardFormatListener(hwnd)
        win32gui.PumpMessages()


class WebsocketServerThread(threading.Thread):
    def __init__(self, read):
        super().__init__()
        self.daemon = True
        self.loop = asyncio.new_event_loop()
        self.read = read
        self.clients = set()

    async def send_text_coroutine(self, text):
        for client in self.clients:
            await client.send(text)

    async def server_handler(self, websocket):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                if self.read and not paused:
                    websocket_queue.put(message)
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
        self.loop.call_soon_threadsafe(self.server.ws_server.close)
        self.loop.call_soon_threadsafe(self.loop.stop)

    def run(self):
        asyncio.set_event_loop(self.loop)
        start_server = websockets.serve(self.server_handler, '0.0.0.0', config.get_general('websocket_port'), max_size=1000000000)
        self.server = start_server
        self.loop.run_until_complete(start_server)
        self.loop.run_forever()
        pending = asyncio.all_tasks(loop=self.loop)
        if len(pending) > 0:
            self.loop.run_until_complete(asyncio.wait(pending))
        self.loop.close()


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
            unixsocket_queue.put(img)
            conn.sendall(b'True')
        else:
            conn.sendall(b'False')


class MacOSWindowTracker(threading.Thread):
    def __init__(self, only_active, window_id):
        super().__init__()
        self.daemon = True
        self.stop = False
        self.only_active = only_active
        self.window_id = window_id
        self.window_x = sct_params['left']
        self.window_y = sct_params['top']
        self.window_width = sct_params['width']
        self.window_height = sct_params['height']
        self.window_active = False
        self.window_minimized = True

    def run(self):
        found = True
        while found and not self.stop:
            found = False
            with objc.autorelease_pool():
                window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, kCGNullWindowID)               
                for i, window in enumerate(window_list):
                    if self.window_id == window['kCGWindowNumber']:
                        found = True
                        bounds = window['kCGWindowBounds']
                        is_minimized = False
                        is_active = window_list[i-1].get(kCGWindowName, '') == 'Dock'
                        break
                if not found:
                    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)               
                    for window in window_list:
                        if self.window_id == window['kCGWindowNumber']:
                            found = True
                            bounds = window['kCGWindowBounds']
                            is_minimized = True
                            is_active = False
                            break
            if bounds['X'] != self.window_x or bounds['Y'] != self.window_y:
                on_window_moved((bounds['X'], bounds['Y']))
                self.window_x = bounds['X']
                self.window_y = bounds['Y']
            if bounds['Width'] != self.window_width or bounds['Height'] != self.window_height:
                on_window_resized((bounds['Width'], bounds['Height']))
                self.window_width = bounds['Width']
                self.window_height = bounds['Height']
            if self.only_active:
                if self.window_active != is_active:
                    on_window_activated(is_active)
                    self.window_active = is_active
            else:
                if self.window_minimized != is_minimized:
                    on_window_minimized(is_minimized)
                    self.window_minimized = is_minimized
            time.sleep(0.2)
        if not found:
            on_window_closed(False)


class TextFiltering:
    accurate_filtering = False

    def __init__(self):
        from pysbd import Segmenter
        self.segmenter = Segmenter(language='ja', clean=True)
        try:
            from transformers import pipeline, AutoTokenizer
            model_ckpt = 'papluca/xlm-roberta-base-language-detection'
            tokenizer = AutoTokenizer.from_pretrained(
                model_ckpt,
                use_fast = False
            )
            self.pipe = pipeline('text-classification', model=model_ckpt, tokenizer=tokenizer)
            self.accurate_filtering = True
        except:
            import langid
            self.classify = langid.classify

    def __call__(self, text, last_text):
        orig_text = self.segmenter.segment(text)
        new_blocks = [block for block in orig_text if block not in last_text]
        final_blocks = []
        if self.accurate_filtering:
            detection_results = self.pipe(new_blocks, top_k=2, truncation=True)
            for idx, block in enumerate(new_blocks):
                if((detection_results[idx][0]['label'] == 'ja' and detection_results[idx][0]['score'] >= 0.85) or
                   (detection_results[idx][1]['label'] == 'ja' and detection_results[idx][1]['score'] >= 0.85)):
                    final_blocks.append(block)
        else:
            for block in new_blocks:
                if self.classify(block)[0] == 'ja':
                    final_blocks.append(block)

        text = '\n'.join(final_blocks)
        return text, orig_text


def pause_handler(is_combo=True):   
    global paused
    global just_unpaused
    if paused:
        message = 'Unpaused!'
        just_unpaused = True
    else:
        message = 'Paused!'

    if is_combo:
        notifier.send_sync(title='owocr', message=message)
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
            notifier.send_sync(title='owocr', message=f'Switched to {new_engine_name}')
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


def on_key_press(key):
    global first_pressed
    if first_pressed == None and key in (keyboard.Key.cmd_l, keyboard.Key.cmd_r, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        first_pressed = key
        pause_handler(False)


def on_key_release(key):
    global first_pressed
    if key == first_pressed:
        pause_handler(False)
        first_pressed = None


def on_screenshot_combo():
    if not paused:
        screenshot_event.set()


def signal_handler(sig, frame):
    global terminated
    logger.info('Terminated!')
    terminated = True


def on_window_closed(alive):
    global terminated
    if not alive:
        logger.info('Window closed, terminated!')
        terminated = True


def on_window_activated(active):
    global screencapture_window_active
    screencapture_window_active = active


def on_window_minimized(minimized):
    global screencapture_window_visible
    screencapture_window_visible = not minimized


def on_window_resized(size):
    global sct_params
    sct_params['width'] = size[0]
    sct_params['height'] = size[1]


def on_window_moved(pos):
    global sct_params
    sct_params['left'] = pos[0]
    sct_params['top'] = pos[1]


async def fix_windows_notifications():
    try:
        await notifier.request_authorisation()
    except OSError:
        pass


def normalize_macos_clipboard(img):
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

    return new_image.TIFFRepresentation()


def are_images_identical(img1, img2):
    if None in (img1, img2):
        return img1 == img2

    img1 = np.array(img1)
    img2 = np.array(img2)

    return (img1.shape == img2.shape) and (img1 == img2).all()


def process_and_write_results(img_or_path, write_to, notifications, enable_filtering, last_text, filtering):
    engine_instance = engine_instances[engine_index]
    t0 = time.time()
    res, text = engine_instance(img_or_path)
    t1 = time.time()

    orig_text = ''
    engine_color = config.get_general('engine_color')
    if res:
        if enable_filtering:
            text, orig_text = filtering(text, last_text)
        text = post_process(text)
        logger.opt(ansi=True).info(f'Text recognized in {t1 - t0:0.03f}s using <{engine_color}>{engine_instance.readable_name}</{engine_color}>: {text}')
        if notifications:
            notifier.send_sync(title='owocr', message='Text recognized: ' + text)

        if write_to == 'websocket':
            websocket_server_thread.send_text(text)
        elif write_to == 'clipboard':
            pyperclipfix.copy(text)
        else:
            with Path(write_to).open('a', encoding='utf-8') as f:
                f.write(text + '\n')
    else:
        logger.opt(ansi=True).info(f'<{engine_color}>{engine_instance.readable_name}</{engine_color}> reported an error after {t1 - t0:0.03f}s: {text}')

    return orig_text


def get_path_key(path):
    return path, path.lstat().st_mtime


def init_config():
    global config
    config = Config()


def run(read_from=None,
        write_to=None,
        engine=None,
        pause_at_startup=None,
        ignore_flag=None,
        delete_images=None,
        notifications=None,
        combo_pause=None,
        combo_engine_switch=None,
        screen_capture_monitor=None,
        screen_capture_coords=None,
        screen_capture_delay_secs=None,
        screen_capture_only_active_windows=None,
        screen_capture_combo=None
        ):
    """
    Japanese OCR client

    Run OCR in the background, waiting for new images to appear either in system clipboard or a directory, or to be sent via a websocket.
    Recognized texts can be either saved to system clipboard, appended to a text file or sent via a websocket.

    :param read_from: Specifies where to read input images from. Can be either "clipboard", "websocket", "unixsocket" (on macOS/Linux), "screencapture", or a path to a directory.
    :param write_to: Specifies where to save recognized texts to. Can be either "clipboard", "websocket", or a path to a text file.
    :param delay_secs: How often to check for new images, in seconds.
    :param engine: OCR engine to use. Available: "mangaocr", "glens", "gvision", "avision", "alivetext", "azure", "winrtocr", "easyocr", "rapidocr".
    :param pause_at_startup: Pause at startup.
    :param ignore_flag: Process flagged clipboard images (images that are copied to the clipboard with the *ocr_ignore* string).
    :param delete_images: Delete image files after processing when reading from a directory.
    :param notifications: Show an operating system notification with the detected text.
    :param combo_pause: Specifies a combo to wait on for pausing the program. As an example: "<ctrl>+<shift>+p". The list of keys can be found here: https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key
    :param combo_engine_switch: Specifies a combo to wait on for switching the OCR engine. As an example: "<ctrl>+<shift>+a". To be used with combo_pause. The list of keys can be found here: https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key
    :param screen_capture_monitor: Specifies monitor to target when reading with screen capture.
    :param screen_capture_coords: Specifies area to target when reading with screen capture. Can be either empty (whole screen), a set of coordinates (x,y,width,height) or a window name (the first matching window title will be used).
    :param screen_capture_delay_secs: Specifies the delay (in seconds) between screenshots when reading with screen capture.
    :param screen_capture_only_active_windows: When reading with screen capture and screen_capture_coords is a window name, specifies whether to only target the window while it's active.
    :param screen_capture_combo: When reading with screen capture, specifies a combo to wait on for taking a screenshot instead of using the delay. As an example: "<ctrl>+<shift>+s". The list of keys can be found here: https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key
    """

    if read_from == 'screencapture' and sys.platform != 'darwin':
        active_window_name = pywinctl.getActiveWindowTitle()

    logger.configure(handlers=[{'sink': sys.stderr, 'format': config.get_general('logger_format')}])

    if config.has_config:
        logger.info('Parsed config file')
    else:
        logger.warning('No config file, defaults will be used.')
        if config.downloaded_config:
            logger.info(f'A default config file has been downloaded to {config.config_path}')

    global engine_instances
    global engine_keys
    engine_instances = []
    config_engines = []
    engine_keys = []
    default_engine = ''

    if len(config.get_general('engines')) > 0:
        for config_engine in config.get_general('engines').split(','):
            config_engines.append(config_engine.strip().lower())

    for _,engine_class in sorted(inspect.getmembers(sys.modules[__name__], lambda x: hasattr(x, '__module__') and x.__module__ and __package__ + '.ocr' in x.__module__ and inspect.isclass(x))):
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
    global notifier
    terminated = False
    paused = pause_at_startup
    just_unpaused = True
    first_pressed = None
    engine_index = engine_keys.index(default_engine) if default_engine != '' else 0
    engine_color = config.get_general('engine_color')
    delay_secs = config.get_general('delay_secs')
    screen_capture_on_combo = False
    notifier = DesktopNotifier()
    key_combos = {}

    if combo_pause != '':
        key_combos[combo_pause] = pause_handler
    if combo_engine_switch != '':
        if combo_pause != '':
            key_combos[combo_engine_switch] = engine_change_handler
        else:
            raise ValueError('combo_pause must also be specified')

    user_input_thread = threading.Thread(target=user_input_thread_run, daemon=True)
    user_input_thread.start()

    if sys.platform == 'win32':
        asyncio.run(fix_windows_notifications())

    if read_from == 'websocket' or write_to == 'websocket':
        global websocket_server_thread
        websocket_server_thread = WebsocketServerThread(read_from == 'websocket')
        websocket_server_thread.start()

    if read_from == 'websocket':
        global websocket_queue
        websocket_queue = queue.Queue()
        read_from_readable = 'websocket'
    elif read_from == 'unixsocket':
        if sys.platform == 'win32':
            raise ValueError('"unixsocket" is not currently supported on Windows')

        global unixsocket_queue
        unixsocket_queue = queue.Queue()
        socket_path = Path('/tmp/owocr.sock')
        if socket_path.exists():
            socket_path.unlink()
        unix_socket_server = socketserver.ThreadingUnixStreamServer(str(socket_path), RequestHandler)
        unix_socket_server_thread = threading.Thread(target=unix_socket_server.serve_forever, daemon=True)
        unix_socket_server_thread.start()
        read_from_readable = 'unix socket'
    elif read_from == 'clipboard':
        mac_clipboard_polling = False
        windows_clipboard_polling = False
        img = None

        if sys.platform == 'darwin':
            from AppKit import NSPasteboard, NSPasteboardTypeTIFF, NSPasteboardTypeString
            pasteboard = NSPasteboard.generalPasteboard()
            count = pasteboard.changeCount()
            mac_clipboard_polling = True
        elif sys.platform == 'win32':
            global clipboard_event
            clipboard_event = threading.Event()
            windows_clipboard_thread = WindowsClipboardThread()
            windows_clipboard_thread.start()
            windows_clipboard_polling = True
        else:
            from PIL import ImageGrab

        read_from_readable = 'clipboard'
    elif read_from == 'screencapture':
        if screen_capture_combo != '':
            screen_capture_on_combo = True
            global screenshot_event
            screenshot_event = threading.Event()
            key_combos[screen_capture_combo] = on_screenshot_combo
        if type(screen_capture_coords) == tuple:
            screen_capture_coords = ','.join(map(str, screen_capture_coords))
        global screencapture_window_active
        global screencapture_window_visible
        screencapture_window_mode = False
        screencapture_window_active = True
        screencapture_window_visible = True
        last_text = []
        sct = mss.mss()
        if screen_capture_coords == '':
            mon = sct.monitors
            if len(mon) <= screen_capture_monitor:
                msg = '"screen_capture_monitor" must be a valid monitor number'
                raise ValueError(msg)
            coord_left = mon[screen_capture_monitor]['left']
            coord_top = mon[screen_capture_monitor]['top']
            coord_width = mon[screen_capture_monitor]['width']
            coord_height = mon[screen_capture_monitor]['height']
        elif len(screen_capture_coords.split(',')) == 4:
            mon = sct.monitors
            if len(mon) <= screen_capture_monitor:
                msg = '"screen_capture_monitor" must be a valid monitor number'
                raise ValueError(msg)
            x, y, coord_width, coord_height = [int(c.strip()) for c in screen_capture_coords.split(',')]
            coord_left = mon[screen_capture_monitor]['left'] + x
            coord_top = mon[screen_capture_monitor]['top'] + y
        else:
            global sct_params
            screencapture_window_mode = True
            if sys.platform == 'darwin':
                window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, kCGNullWindowID)
                window_titles = []
                window_id = 0
                after_dock = False
                target_title = None
                for i, window in enumerate(window_list):
                    window_title = window.get(kCGWindowName, '')
                    if after_dock and psutil.Process(window['kCGWindowOwnerPID']).name() not in ('Terminal', 'iTerm2'):
                        window_titles.append(window_title)
                    if window_title == 'Dock':
                        after_dock = True

                if screen_capture_coords in window_titles:
                    target_title = screen_capture_coords
                else:
                    for t in window_titles:
                        if screen_capture_coords in t:
                            target_title = t
                            break

                if not target_title:
                    msg = '"screen_capture_coords" must be empty (for the whole screen), a valid set of coordinates, or a valid window name'
                    raise ValueError(msg)

                for i, window in enumerate(window_list):
                    window_title = window.get(kCGWindowName, '')
                    if target_title == window_title:
                        window_id = window['kCGWindowNumber']
                        bounds = window['kCGWindowBounds']
                        break

                if screen_capture_only_active_windows:
                    screencapture_window_active = False
                else:
                    screencapture_window_visible = False
                sct_params = {'top': bounds['Y'], 'left': bounds['X'], 'width': bounds['Width'], 'height': bounds['Height']}
                macos_window_tracker = MacOSWindowTracker(screen_capture_only_active_windows, window_id)
                macos_window_tracker.start()
            else:
                window_title = None
                window_titles = pywinctl.getAllTitles()
                if screen_capture_coords in window_titles:
                    window_title = screen_capture_coords
                else:
                    for t in window_titles:
                        if screen_capture_coords in t and t != active_window_name:
                            window_title = t
                            break

                if not window_title:
                    msg = '"screen_capture_coords" must be empty (for the whole screen), a valid set of coordinates, or a valid window name'
                    raise ValueError(msg)

                target_window = pywinctl.getWindowsWithTitle(window_title)[0]
                coord_top = target_window.top
                coord_left = target_window.left
                coord_width = target_window.width
                coord_height = target_window.height
                sct_params = {'top': coord_top, 'left': coord_left, 'width': coord_width, 'height': coord_height}
                if screen_capture_only_active_windows:
                    screencapture_window_active = target_window.isActive
                    target_window.watchdog.start(isAliveCB=on_window_closed, isActiveCB=on_window_activated, resizedCB=on_window_resized, movedCB=on_window_moved)
                else:
                    screencapture_window_visible = not target_window.isMinimized
                    target_window.watchdog.start(isAliveCB=on_window_closed, isMinimizedCB=on_window_minimized, resizedCB=on_window_resized, movedCB=on_window_moved)

        filtering = TextFiltering()
        read_from_readable = 'screen capture'
    else:
        read_from = Path(read_from)
        if not read_from.is_dir():
            raise ValueError('read_from must be either "websocket", "unixsocket", "clipboard", "screencapture", or a path to a directory')

        allowed_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
        old_paths = set()
        for path in read_from.iterdir():
            if path.suffix.lower() in allowed_extensions:
                old_paths.add(get_path_key(path))

        read_from_readable = f'directory {read_from}'

    if len(key_combos) > 0:
        key_combo_listener = keyboard.GlobalHotKeys(key_combos)
    else:
        key_combo_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    key_combo_listener.start()

    if write_to in ('clipboard', 'websocket'):
        write_to_readable = write_to
    else:
        if Path(write_to).suffix.lower() != '.txt':
            raise ValueError('write_to must be either "websocket", "clipboard" or a path to a text file')
        write_to_readable = f'file {write_to}'
    logger.opt(ansi=True).info(f"Reading from {read_from_readable}, writing to {write_to_readable} using <{engine_color}>{engine_instances[engine_index].readable_name}</{engine_color}>{' (paused)' if paused else ''}")
    signal.signal(signal.SIGINT, signal_handler)

    while not terminated:
        if read_from == 'websocket':
            while True:
                try:
                    item = websocket_queue.get(timeout=delay_secs)
                except queue.Empty:
                    break
                else:
                    if not paused:
                        img = Image.open(io.BytesIO(item))
                        process_and_write_results(img, write_to, notifications, False, '', None)
        elif read_from == 'unixsocket':
            while True:
                try:
                    item = unixsocket_queue.get(timeout=delay_secs)
                except queue.Empty:
                    break
                else:
                    if not paused:
                        img = Image.open(io.BytesIO(item))
                        process_and_write_results(img, write_to, notifications, False, '', None)
        elif read_from == 'clipboard':
            process_clipboard = False
            if windows_clipboard_polling:
                if clipboard_event.wait(delay_secs):
                    clipboard_event.clear()
                    while True:
                        try:
                            win32clipboard.OpenClipboard()
                            break
                        except pywintypes.error:
                            pass
                        time.sleep(0.1)
                    if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_DIB):
                        clipboard_text = ''
                        if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
                            clipboard_text = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
                        if ignore_flag or clipboard_text != '*ocr_ignore*':
                            img = Image.open(io.BytesIO(win32clipboard.GetClipboardData(win32clipboard.CF_DIB)))
                            process_clipboard = True
                    win32clipboard.CloseClipboard()
            elif mac_clipboard_polling:
                if not paused:
                    with objc.autorelease_pool():
                        old_count = count
                        count = pasteboard.changeCount()
                        if not just_unpaused and count != old_count and NSPasteboardTypeTIFF in pasteboard.types():
                            clipboard_text = ''
                            if NSPasteboardTypeString in pasteboard.types():
                                clipboard_text = pasteboard.stringForType_(NSPasteboardTypeString)
                            if ignore_flag or clipboard_text != '*ocr_ignore*':
                                img = normalize_macos_clipboard(pasteboard.dataForType_(NSPasteboardTypeTIFF))
                                img = Image.open(io.BytesIO(img))
                                process_clipboard = True
            else:
                if not paused:
                    old_img = img
                    try:
                        img = ImageGrab.grabclipboard()
                    except Exception:
                        pass
                    else:
                        if ((not just_unpaused) and isinstance(img, Image.Image) and \
                            (ignore_flag or pyperclipfix.paste() != '*ocr_ignore*') and \
                            (not are_images_identical(img, old_img))):
                            process_clipboard = True

            if process_clipboard:
                process_and_write_results(img, write_to, notifications, False, '', None)

            just_unpaused = False

            if not windows_clipboard_polling:
                time.sleep(delay_secs)
        elif read_from == 'screencapture':
            if screen_capture_on_combo:
                take_screenshot = screenshot_event.wait(delay_secs)
                if take_screenshot:
                    screenshot_event.clear()
            else:
                take_screenshot = screencapture_window_active and not paused

            if take_screenshot and screencapture_window_visible:
                sct_img = sct.grab(sct_params)
                img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
                res = process_and_write_results(img, write_to, notifications, True, last_text, filtering)
                if res != '':
                    last_text = res
                delay = screen_capture_delay_secs
            else:
                delay = delay_secs

            if not screen_capture_on_combo:
                time.sleep(delay)
        else:
            for path in read_from.iterdir():
                if path.suffix.lower() in allowed_extensions:
                    path_key = get_path_key(path)
                    if path_key not in old_paths:
                        old_paths.add(path_key)

                        if not paused:
                            try:
                                img = Image.open(path)
                                img.load()
                            except (UnidentifiedImageError, OSError) as e:
                                logger.warning(f'Error while reading file {path}: {e}')
                            else:
                                process_and_write_results(img, write_to, notifications, False, '', None)
                                img.close()
                                if delete_images:
                                    Path.unlink(path)

            time.sleep(delay_secs)

    if read_from == 'websocket' or write_to == 'websocket':
        websocket_server_thread.stop_server()
        websocket_server_thread.join()
    if read_from == 'clipboard' and windows_clipboard_polling:
        win32api.PostThreadMessage(windows_clipboard_thread.thread_id, win32con.WM_QUIT, 0, 0)
        windows_clipboard_thread.join()
    elif read_from == 'screencapture' and screencapture_window_mode:
        if sys.platform == 'darwin':
            macos_window_tracker.stop = True
            macos_window_tracker.join()
        else:
            target_window.watchdog.stop()
    elif read_from == 'unixsocket':
        unix_socket_server.shutdown()
        unix_socket_server_thread.join()
    key_combo_listener.stop()
