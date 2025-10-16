import os
import configparser
import argparse
import textwrap
import urllib.request

def str2bool(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(prog='owocr', description=textwrap.dedent('''\
    Runs OCR in the background.
    It can read images copied to the system clipboard or placed in a directory, images sent via a websocket or a Unix domain socket, or directly capture a screen (or a portion of it) or a window.
    Recognized text can be either saved to system clipboard, appended to a text file or sent via a websocket.
'''))

parser.add_argument('-r', '--read_from', type=str, default=argparse.SUPPRESS,
                    help='Where to read input images from. Can be either "clipboard", "websocket", "unixsocket" (on macOS/Linux), "screencapture", or a path to a directory.')
parser.add_argument('-rs', '--read_from_secondary', type=str, default=argparse.SUPPRESS,
                    help="Optional secondary source to read input images from. Same options as read_from, but they can't both be directory paths.")
parser.add_argument('-w', '--write_to', type=str, default=argparse.SUPPRESS,
                    help='Where to save recognized texts to. Can be either "clipboard", "websocket", or a path to a text file.')
parser.add_argument('-e', '--engine', type=str, default=argparse.SUPPRESS,
                    help='OCR engine to use. Available: "mangaocr", "glens", "bing", "gvision", "avision", "alivetext", "azure", "winrtocr", "oneocr", "easyocr", "rapidocr", "ocrspace".')
parser.add_argument('-es', '--engine_secondary', type=str, default=argparse.SUPPRESS,
                    help='OCR engine to use for two-pass processing.')
parser.add_argument('-p', '--pause_at_startup', type=str2bool, nargs='?', const=True, default=argparse.SUPPRESS,
                    help='Pause at startup.')
parser.add_argument('-d', '--delete_images', type=str2bool, nargs='?', const=True, default=argparse.SUPPRESS,
                    help='Delete image files after processing when reading from a directory.')
parser.add_argument('-n', '--notifications', type=str2bool, nargs='?', const=True, default=argparse.SUPPRESS,
                    help='Show an operating system notification with the detected text. Will be ignored when reading with screen capture and periodic screenshots.')
parser.add_argument('-a', '--auto_pause', type=float, default=argparse.SUPPRESS,
                    help='Automatically pause the program after the specified amount of seconds since the last successful text recognition. 0 to disable.')
parser.add_argument('-cp', '--combo_pause', type=str, default=argparse.SUPPRESS,
                    help='Combo to wait on for pausing the program. As an example: "<ctrl>+<shift>+p". The list of keys can be found here: https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key')
parser.add_argument('-cs', '--combo_engine_switch', type=str, default=argparse.SUPPRESS,
                    help='Combo to wait on for switching the OCR engine. As an example: "<ctrl>+<shift>+a". The list of keys can be found here: https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key')
parser.add_argument('-sa', '--screen_capture_area', type=str, default=argparse.SUPPRESS,
                    help='Area to target when reading with screen capture. Can be either empty (automatic selector), a set of coordinates (x,y,width,height), "screen_N" (captures a whole screen, where N is the screen number starting from 1) or a window name (the first matching window title will be used).')
parser.add_argument('-swa', '--screen_capture_window_area', type=str, default=argparse.SUPPRESS,
                    help='If capturing with screen capture, subsection of the selected window. Can be either empty (automatic selector), a set of coordinates (x,y,width,height), "window" to use the whole window.')
parser.add_argument('-sd', '--screen_capture_delay_secs', type=float, default=argparse.SUPPRESS,
                    help='Delay (in seconds) between screenshots when reading with screen capture. -1 to disable periodic screenshots.')
parser.add_argument('-sw', '--screen_capture_only_active_windows', type=str2bool, nargs='?', const=True, default=argparse.SUPPRESS,
                    help="When reading with screen capture and screen_capture_area is a window name, only target the window while it's active.")
parser.add_argument('-sf', '--screen_capture_frame_stabilization', type=float, default=argparse.SUPPRESS,
                    help="When reading with screen capture, delay to wait until text is stable before processing it. -1 waits for two OCR results to be the same. 0 to disable.")
parser.add_argument('-sl', '--screen_capture_line_recovery', type=str2bool, nargs='?', const=True, default=argparse.SUPPRESS,
                    help="When reading with screen capture and frame stabilization is on, try to recover missed lines from unstable frames. Can lead to increased glitches.")
parser.add_argument('-sc', '--screen_capture_combo', type=str, default=argparse.SUPPRESS,
                    help='When reading with screen capture, combo to wait on for taking a screenshot. If periodic screenshots are also enabled, any screenshot taken this way bypasses the filtering. Example value: "<ctrl>+<shift>+s". The list of keys can be found here: https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key')
parser.add_argument('-scc', '--coordinate_selector_combo', type=str, default=argparse.SUPPRESS,
                    help='When reading with screen capture, combo to wait on for invoking the coordinate picker to change the screen/window area. Example value: "<ctrl>+<shift>+c". The list of keys can be found here: https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key')
parser.add_argument('-l', '--language', type=str, default=argparse.SUPPRESS,
                    help='Two letter language code for filtering screencapture OCR results. Ex. "ja" for Japanese, "zh" for Chinese, "ko" for Korean, "ar" for Arabic, "ru" for Russian, "el" for Greek, "he" for Hebrew, "th" for Thai. Any other value will use Latin Extended (for most European languages and English).')
parser.add_argument('-f', '--furigana_filter', type=str2bool, nargs='?', const=True, default=argparse.SUPPRESS,
                    help="Try to filter furigana lines for Japanese.")
parser.add_argument('-of', '--output_format', type=str, default=argparse.SUPPRESS,
                    help='The output format for OCR results. Can be "text" (default) or "json" (to include coordinates).')
parser.add_argument('-wp', '--websocket_port', type=int, default=argparse.SUPPRESS,
                    help='Websocket port to use if reading or writing to websocket.')
parser.add_argument('-ds', '--delay_seconds', type=float, default=argparse.SUPPRESS,
                    help='Delay (in seconds) between checks when reading from clipboard (on macOS/Linux) or a directory.')
parser.add_argument('-v', '--verbosity', type=int, default=argparse.SUPPRESS,
                    help='Terminal window verbosity. Can be -2 (all recognized text is showed whole, default), -1 (only timestamps are shown), 0 (nothing is shown but errors), or larger than 0 to cut displayed text to that amount of characters.')
parser.add_argument('--uwu', type=str2bool, nargs='?', const=True, default=argparse.SUPPRESS, help=argparse.SUPPRESS)

class Config:
    has_config = False
    downloaded_config = False
    config_path = os.path.join(os.path.expanduser('~'),'.config','owocr_config.ini')
    __general_config = {}
    __engine_config = {}
    __default_config = {
        'read_from': 'clipboard',
        'read_from_secondary': '',
        'write_to': 'clipboard',
        'engine': '',
        'engine_secondary': '',
        'pause_at_startup': False,
        'auto_pause' : 0,
        'delete_images': False,
        'engines': [],
        'logger_format': '<green>{time:HH:mm:ss.SSS}</green> | <level>{message}</level>',
        'engine_color': 'cyan',
        'delay_secs': 0.5,
        'websocket_port': 7331,
        'notifications': False,
        'combo_pause': '',
        'combo_engine_switch': '',
        'screen_capture_area': '',
        'screen_capture_window_area': 'window',
        'screen_capture_delay_secs': 0,
        'screen_capture_only_active_windows': True,
        'screen_capture_frame_stabilization': -1,
        'screen_capture_line_recovery': True,
        'furigana_filter': True,
        'screen_capture_combo': '',
        'coordinate_selector_combo': '',
        'screen_capture_old_macos_api': False,
        'language': 'ja',
        'output_format': 'text',
        'verbosity': -2,
        'uwu': False
    }

    def __parse(self, value):
        value = value.strip()
        if value.lower() == 'false':
            return False
        if value.lower() == 'true':
            return True
        try:
            int(value)
            return int(value)
        except ValueError:
            pass
        try:
            float(value)
            return float(value)
        except ValueError:
            pass
        return value

    def __init__(self):
        args = parser.parse_args()
        self.__provided_cli_args = vars(args)
        config = configparser.ConfigParser()
        res = config.read(self.config_path)

        if len(res) == 0:
            try:
                config_folder = os.path.join(os.path.expanduser('~'),'.config')
                if not os.path.isdir(config_folder):
                    os.makedirs(config_folder)
                urllib.request.urlretrieve('https://github.com/AuroraWright/owocr/raw/master/owocr_config.ini', self.config_path)
                self.downloaded_config = True
            finally:
                return

        self.has_config = True
        for key in config:
            if key == 'general':
                for sub_key in config[key]:
                    self.__general_config[sub_key.lower()] = self.__parse(config[key][sub_key])
            elif key != 'DEFAULT':
                self.__engine_config[key.lower()] = {}
                for sub_key in config[key]:
                    self.__engine_config[key.lower()][sub_key.lower()] = self.__parse(config[key][sub_key])

    def get_general(self, value):
        if self.__provided_cli_args.get(value, None) is not None:
            return self.__provided_cli_args[value]
        try:
            return self.__general_config[value]
        except KeyError:
            if value in self.__default_config:
                return self.__default_config[value]
            else:
                return None

    def get_engine(self, value):
        try:
            return self.__engine_config[value]
        except KeyError:
            return None

config = Config()
