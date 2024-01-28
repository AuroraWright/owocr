import os
import configparser

class Config:
    has_config = False
    __general_config = {}
    __engine_config = {}
    __default_config = {
        'read_from': 'clipboard',
        'write_to': 'clipboard',
        'engine': '',
        'pause_at_startup': False,
        'ignore_flag': False,
        'delete_images': False,
        'engines': [],
        'logger_format': '<green>{time:HH:mm:ss.SSS}</green> | <level>{message}</level>',
        'engine_color': 'cyan',
        'delay_secs': 0.5,
        'websocket_port': 7331,
        'notifications': False,
        'screen_capture_monitor': 1,
        'screen_capture_coords': '',
        'screen_capture_delay_secs': 3
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
        config_file = os.path.join(os.path.expanduser('~'),'.config','owocr_config.ini')
        config = configparser.ConfigParser()
        res = config.read(config_file)

        if len(res) != 0:
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