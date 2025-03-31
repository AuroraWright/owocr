import os
import configparser
import urllib.request


class Config:
    has_config = False
    downloaded_config = False
    config_path = os.path.join(os.path.expanduser('~'),'.config','owocr_config.ini')
    __general_config = {}
    __engine_config = {}
    __default_config = {
        'read_from': 'clipboard',
        'write_to': 'clipboard',
        'engine': '',
        'pause_at_startup': False,
        'auto_pause' : 0,
        'ignore_flag': False,
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
        'screen_capture_delay_secs': 3,
        'screen_capture_only_active_windows': True,
        'screen_capture_combo': ''
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