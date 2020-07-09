from .loggers.loggerRegistry import LOGGERS

from .loggers.base_logger import *


class Loggers():
    def __init__(self, logger_configs: dict):
        assert isinstance(logger_configs, dict), '%s logger config should be a dict' % str(logger_configs)
        self.loggers = {}
        print(logger_configs)  # TODO: for debug
        for key, value in logger_configs.items():
            logtype = LOGGERS.get(key)
            if logtype is None:
                raise NotImplementedError(key)
            self.loggers[key] = logtype(**value)

    def update_loss(self, info, shouldprint):  # calculate sum
        for key, value in self.loggers.items():
            # print('logger update_loss', key, value)
            value.update_loss(info, shouldprint)

    def update_error(self, info, shouldprint):
        for key, value in self.loggers.items():
            # print('logger update_error', key, value)
            value.update_error(info, shouldprint)
