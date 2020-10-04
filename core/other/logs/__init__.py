import os
import importlib
basename = 'base_logger'


class Loggers():
    def __init__(self, logger_configs: dict):
        assert isinstance(logger_configs, dict), '%s logger config should be a dict' % str(logger_configs)
        self.loggers = {}
        print(logger_configs)  # TODO: for debug
        for key, value in logger_configs.items():
            relative_key = '.' + key
            print('relatively import', key, 'from core.other.logs.loggers')
            try: 
                # logtype = importlib.import_module(relative_key, 'core.other.logs.loggers')
                logtype = importlib.import_module('core.other.logs.loggers' + relative_key)
            except Exception as e:
                print('logger', key, 'not exist')
                print(str(e))
                raise e
            # import core.other.logs.loggers.base_logger
            # logtype = importlib.import_module('core.other.logs.loggers.%s' % key)
            if key != basename:
                assert basename in logger_configs.keys(), 'should calculate float value in base_logger'
            self.loggers[key] = logtype.logger(**value)

    def update_loss(self, info, shouldprint):  # calculate sum
        if basename in self.loggers:
            self.loggers[basename].update_loss(info, shouldprint)
            floatvalue = self.loggers[basename].get_float_value()
        for key, value in self.loggers.items():
            if key == basename:
                continue
            value.update(info, floatvalue, shouldprint, 'loss')

    def update_error(self, info, shouldprint):
        if basename in self.loggers:
            self.loggers[basename].update_error(info, shouldprint)
            floatvalue = self.loggers[basename].get_float_value()
        for key, value in self.loggers.items():
            if key == basename:
                continue
            value.update(info, floatvalue, shouldprint, 'error')
