import sys
import time
import threading
import os
import configparser
from pathlib import Path

import fire
import numpy as np
import pyperclip
from PIL import Image
from PIL import UnidentifiedImageError
from loguru import logger
from pynput import keyboard

import inspect
from manga_ocr import *


def are_images_identical(img1, img2):
    if None in (img1, img2):
        return img1 == img2

    img1 = np.array(img1)
    img2 = np.array(img2)

    return (img1.shape == img2.shape) and (img1 == img2).all()


def process_and_write_results(engine_instance, img_or_path, write_to):
    t0 = time.time()
    text = engine_instance(img_or_path)
    t1 = time.time()

    logger.opt(ansi=True).info(f"Text recognized in {t1 - t0:0.03f}s using <cyan>{engine_instance.readable_name}</cyan>: {text}")

    if write_to == 'clipboard':
        pyperclip.copy(text)
    else:
        write_to = Path(write_to)
        if write_to.suffix != '.txt':
            raise ValueError('write_to must be either "clipboard" or a path to a text file')

        with write_to.open('a', encoding="utf-8") as f:
            f.write(text + '\n')


def get_path_key(path):
    return path, path.lstat().st_mtime


def getchar_thread():
    global user_input
    if sys.platform == "win32":
        import msvcrt
        while True:
            user_input = msvcrt.getch()
            if user_input.lower() in 'tq':
                break
    else:
        import tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                user_input = sys.stdin.read(1)
                if user_input.lower() in 'tq':
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def on_key_press(key):
    global tmp_paused
    if key == keyboard.Key.cmd_r or key == keyboard.Key.ctrl_r:
        tmp_paused = True


def on_key_release(key):
    global tmp_paused
    global just_unpaused
    if key == keyboard.Key.cmd_r or key == keyboard.Key.ctrl_r:
        tmp_paused = False
        just_unpaused = True


def run(read_from='clipboard',
        write_to='clipboard',
        delay_secs=0.5,
        engine='',
        pause_at_startup=False,
        ignore_flag=False,
        verbose=False
        ):
    """
    Run OCR in the background, waiting for new images to appear either in system clipboard, or a directory.
    Recognized texts can be either saved to system clipboard, or appended to a text file.

    :param read_from: Specifies where to read input images from. Can be either "clipboard", or a path to a directory.
    :param write_to: Specifies where to save recognized texts to. Can be either "clipboard", or a path to a text file.
    :param delay_secs: How often to check for new images, in seconds.
    :param engine: OCR engine to use. Available: "mangaocr", "gvision", "avision", "azure", "winrtocr", "easyocr", "paddleocr".
    :param pause_at_startup: Pause at startup.
    :param ignore_flag: Process flagged images (images that are copied to the clipboard with the *ocr_ignore* string).
    :param verbose: If True, unhides all warnings.
    """

    fmt = "<green>{time:HH:mm:ss.SSS}</green> | <level>{message}</level>"
    config = {
        "handlers": [
            {"sink": sys.stderr, "format": fmt},
        ],
    }
    logger.configure(**config)

    if sys.platform not in ('darwin', 'win32') and write_to == 'clipboard':
        # Check if the system is using Wayland
        if os.environ.get('WAYLAND_DISPLAY'):
            # Check if the wl-clipboard package is installed
            if os.system("which wl-copy > /dev/null") == 0:
                pyperclip.set_clipboard("wl-clipboard")
            else:
                msg = 'Your session uses wayland and does not have wl-clipboard installed. ' \
                    'Install wl-clipboard for write in clipboard to work.'
                raise NotImplementedError(msg)

    engine_instances = []
    config_engines = []
    engine_keys = []
    default_engine = ''

    logger.info(f'Parsing config file')
    config_file = os.path.join(os.path.expanduser('~'),'.config','ocr_config.ini')
    config = configparser.ConfigParser()
    res = config.read(config_file)

    if len(res) == 0:
        logger.warning('No config file, defaults will be used')
    else:
        try:
            for config_engine in config['common']['engines'].split(','):
                config_engines.append(config_engine.strip())
        except KeyError:
            pass

    for _,engine_class in sorted(inspect.getmembers(sys.modules[__name__], lambda x: hasattr(x, '__module__') and __package__ in x.__module__ and inspect.isclass(x))):
        if len(config_engines) == 0 or engine_class.name in config_engines:
            try:
                engine_instance = engine_class(config[engine_class.name])
            except KeyError:
                engine_instance = engine_class()

            if engine_instance.available:
                engine_instances.append(engine_instance)
                engine_keys.append(engine_class.key)
                if engine == engine_class.name:
                    default_engine = engine_class.key

    if len(engine_keys) == 0:
        msg = 'No engines available!'
        raise NotImplementedError(msg)

    engine_index = engine_keys.index(default_engine) if default_engine != '' else 0

    global user_input
    user_input = ''

    user_input_thread = threading.Thread(target=getchar_thread, daemon=True)
    user_input_thread.start()

    if read_from == 'clipboard':
        from PIL import ImageGrab

        global just_unpaused
        global tmp_paused
        paused = pause_at_startup
        just_unpaused = True
        tmp_paused = False
        img = None

        logger.opt(ansi=True).info(f"Reading from clipboard using <cyan>{engine_instances[engine_index].readable_name}</cyan>{' (paused)' if paused else ''}")

        if sys.platform == "darwin" and 'objc' in sys.modules:
            from AppKit import NSPasteboard, NSPasteboardTypePNG, NSPasteboardTypeTIFF
            pasteboard = NSPasteboard.generalPasteboard()
            count = pasteboard.changeCount()
            mac_clipboard_polling = True
        else:
            mac_clipboard_polling = False

        tmp_paused_listener = keyboard.Listener(
            on_press=on_key_press,
            on_release=on_key_release)
        tmp_paused_listener.start()
    else:
        read_from = Path(read_from)
        if not read_from.is_dir():
            raise ValueError('read_from must be either "clipboard" or a path to a directory')

        logger.opt(ansi=True).info(f'Reading from directory {read_from} using <cyan>{engine_instances[engine_index].readable_name}</cyan>')

        old_paths = set()
        for path in read_from.iterdir():
            old_paths.add(get_path_key(path))

    while True:
        if user_input != '':
            if user_input.lower() in 'tq':
                if read_from == 'clipboard':
                    tmp_paused_listener.stop()
                user_input_thread.join()
                logger.info('Terminated!')
                break

            new_engine_index = engine_index

            if read_from == 'clipboard' and user_input.lower() == 'p':
                if paused:
                    logger.info('Unpaused!')
                    just_unpaused = True
                else:
                    logger.info('Paused!')
                paused = not paused
            elif user_input.lower() == 's':
                if engine_index == len(engine_keys) - 1:
                    new_engine_index = 0
                else:
                    new_engine_index = engine_index + 1
            elif user_input.lower() in engine_keys:
                new_engine_index = engine_keys.index(user_input.lower())

            if engine_index != new_engine_index:
                engine_index = new_engine_index
                logger.opt(ansi=True).info(f"Switched to <cyan>{engine_instances[engine_index].readable_name}</cyan>!")

            user_input = ''

        if read_from == 'clipboard':
            if not paused and not tmp_paused:
                if mac_clipboard_polling:
                    old_count = count
                    count = pasteboard.changeCount()
                    changed = not just_unpaused and count != old_count and any(x in pasteboard.types() for x in [NSPasteboardTypePNG, NSPasteboardTypeTIFF])
                else:
                    changed = True

                if changed:
                    old_img = img

                    try:
                        img = ImageGrab.grabclipboard()
                    except OSError as error:
                        if not verbose and "cannot identify image file" in str(error):
                            # Pillow error when clipboard hasn't changed since last grab (Linux)
                            pass
                        elif not verbose and "target image/png not available" in str(error):
                            # Pillow error when clipboard contains text (Linux, X11)
                            pass
                        else:
                            logger.warning('Error while reading from clipboard ({})'.format(error))
                    else:
                        if not just_unpaused and (ignore_flag or pyperclip.paste() != '*ocr_ignore*') and isinstance(img, Image.Image) and not are_images_identical(img, old_img):
                            process_and_write_results(engine_instances[engine_index], img, write_to)

            if just_unpaused:
                just_unpaused = False
        else:
            for path in read_from.iterdir():
                path_key = get_path_key(path)
                if path_key not in old_paths:
                    old_paths.add(path_key)

                    try:
                        img = Image.open(path)
                        img.load()
                    except (UnidentifiedImageError, OSError) as e:
                        logger.warning(f'Error while reading file {path}: {e}')
                    else:
                        process_and_write_results(engine_instances[engine_index], img, write_to)

        time.sleep(delay_secs)

if __name__ == '__main__':
    fire.Fire(run)
