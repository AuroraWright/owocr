import sys
import time
import threading
import os
from pathlib import Path

import fire
import numpy as np
import pyperclip
from PIL import Image
from PIL import UnidentifiedImageError
from loguru import logger
from pynput import keyboard

from manga_ocr import MangaOcr
from manga_ocr import GoogleVision
from manga_ocr import AppleVision
from manga_ocr import AzureComputerVision

engines = ['avision', 'gvision', 'azure', 'mangaocr']


def get_engine_name(engine):
    engine_names = ['Apple Vision', 'Google Vision', 'Azure Computer Vision', 'Manga OCR']
    return engine_names[engines.index(engine)]


def are_images_identical(img1, img2):
    if None in (img1, img2):
        return img1 == img2

    img1 = np.array(img1)
    img2 = np.array(img2)

    return (img1.shape == img2.shape) and (img1 == img2).all()


def process_and_write_results(mocr, avision, gvision, azure, img_or_path, write_to, engine):
    t0 = time.time()
    if engine == 'gvision':
        text = gvision(img_or_path)
    elif engine == 'avision':
        text = avision(img_or_path)
    elif engine == 'azure':
        text = azure(img_or_path)
    else:
        text = mocr(img_or_path)
    t1 = time.time()

    logger.opt(ansi=True).info(f"Text recognized in {t1 - t0:0.03f}s using <cyan>{get_engine_name(engine)}</cyan>: {text}")

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


def run(read_from='clipboard',
        write_to='clipboard',
        pretrained_model_name_or_path='kha-white/manga-ocr-base',
        force_cpu=False,
        delay_secs=0.5,
        engine='mangaocr',
        verbose=False
        ):
    """
    Run OCR in the background, waiting for new images to appear either in system clipboard, or a directory.
    Recognized texts can be either saved to system clipboard, or appended to a text file.

    :param read_from: Specifies where to read input images from. Can be either "clipboard", or a path to a directory.
    :param write_to: Specifies where to save recognized texts to. Can be either "clipboard", or a path to a text file.
    :param pretrained_model_name_or_path: Path to a trained model, either local or from Transformers' model hub.
    :param force_cpu: If True, OCR will use CPU even if GPU is available.
    :param delay_secs: How often to check for new images, in seconds.
    :param engine: OCR engine to use. Available: "mangaocr", "gvision", "avision", "azure".
    :param verbose: If True, unhides all warnings.
    """

    fmt = "<green>{time:HH:mm:ss.SSS}</green> | <level>{message}</level>"
    config = {
        "handlers": [
            {"sink": sys.stderr, "format": fmt},
        ],
    }
    logger.configure(**config)

    mocr = MangaOcr(pretrained_model_name_or_path, force_cpu)
    gvision = GoogleVision()
    azure = AzureComputerVision()
    avision = AppleVision()

    if engine not in engines:
        msg = 'Unknown OCR engine!'
        raise NotImplementedError(msg)

    if sys.platform not in ('darwin', 'win32') and write_to == 'clipboard':
        # Check if the system is using Wayland
        import os
        if os.environ.get('WAYLAND_DISPLAY'):
            # Check if the wl-clipboard package is installed
            if os.system("which wl-copy > /dev/null") == 0:
                pyperclip.set_clipboard("wl-clipboard")
            else:
                msg = 'Your session uses wayland and does not have wl-clipboard installed. ' \
                    'Install wl-clipboard for write in clipboard to work.'
                raise NotImplementedError(msg)

    if read_from == 'clipboard':
        from PIL import ImageGrab
        logger.info('Reading from clipboard')

        paused = False
        global just_unpaused
        just_unpaused = True
        img = None

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

        global tmp_paused
        tmp_paused = False

        tmp_paused_listener = keyboard.Listener(
            on_press=on_key_press,
            on_release=on_key_release)
        tmp_paused_listener.start()
    else:
        read_from = Path(read_from)
        if not read_from.is_dir():
            raise ValueError('read_from must be either "clipboard" or a path to a directory')

        logger.info(f'Reading from directory {read_from}')

        old_paths = set()
        for path in read_from.iterdir():
            old_paths.add(get_path_key(path))

    def getchar_thread():
        global user_input
        import os
        if os.name == 'nt': # how it works on windows
            import msvcrt
            while True:
                user_input = msvcrt.getch()
                if user_input.lower() in 'tq':
                    break
        else:
            import tty, termios, sys
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

    global user_input
    user_input = ''

    user_input_thread = threading.Thread(target=getchar_thread, daemon=True)
    user_input_thread.start()

    while True:
        if user_input != '':
            if user_input.lower() in 'tq':
                if read_from == 'clipboard':
                    tmp_paused_listener.stop()
                user_input_thread.join()
                logger.info('Terminated!')
                break
            if read_from == 'clipboard' and user_input.lower() == 'p':
                if paused:
                    logger.info('Unpaused!')
                    just_unpaused = True
                else:
                    logger.info('Paused!')
                paused = not paused
            elif user_input.lower() == 's':
                if engine == engines[-1]:
                    engine = engines[0]
                else:
                    engine = engines[engines.index(engine) + 1]

                logger.opt(ansi=True).info(f"Switched to <cyan>{get_engine_name(engine)}</cyan>!")
            elif user_input.lower() in 'agvm':
                new_engine = engines['agvm'.find(user_input.lower())]
                if engine != new_engine:
                    engine = new_engine
                    logger.opt(ansi=True).info(f"Switched to <cyan>{get_engine_name(engine)}</cyan>!")

            user_input = ''

        if read_from == 'clipboard':
            if not paused and not tmp_paused:
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
                    if not just_unpaused and isinstance(img, Image.Image) and not are_images_identical(img, old_img):
                        process_and_write_results(mocr, avision, gvision, azure, img, write_to, engine)

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
                        process_and_write_results(mocr, avision, gvision, azure, img, write_to, engine)

        time.sleep(delay_secs)

if __name__ == '__main__':
    fire.Fire(run)
