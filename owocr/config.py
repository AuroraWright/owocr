import os
import configparser

class Config:
    has_config = False
    general_config = {}
    engine_config = {}

    def _parse(self, value):
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
                        self.general_config[sub_key.lower()] = self._parse(config[key][sub_key])
                elif key != 'DEFAULT':
                    self.engine_config[key.lower()] = {}
                    for sub_key in config[key]:
                        self.engine_config[key.lower()][sub_key.lower()] = self._parse(config[key][sub_key])

    def get_general(self, value):
        try:
            return self.general_config[value]
        except KeyError:
            return None

    def get_engine(self, value):
        try:
            return self.engine_config[value]
        except KeyError:
            return None