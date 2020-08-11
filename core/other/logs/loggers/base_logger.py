from core.utils.utils import create_logger
from .loggerRegistry import LOGGERS
import numpy as np
import torch


@LOGGERS.register_module()
class baselogger():
    def __init__(self, path):
        self.logger = create_logger('base', path)
        self.info = {'loss': {}, 'error': {}}

    def _direct_print(self, info, keyword: str):
        for key, value in info.items():
            if keyword in key:
                self.logger.info("%s:%s" % (key, str(value)))

    def __error_div_count(self, info, key, value):
        '''divide: *(error) /= *(error_count); if not exist /=n_count'''
        divide = None
        if 'n_count' in info.keys():
            divide = info['n_count']  # mean of this iteration
        if '(error)' in key:
            div_name = key.replace('(error)', '(error_count)')
            if div_name in info.keys():
                divide = info[div_name]
        if divide is not None:
            value = value / divide  # mean of this iteration
        else:
            raise NotImplementedError('Error fac name not in keyword; error_key = %s' % key)
        return value

    def __to_string(self, value, all=True):
        if isinstance(value, float):
            return '%.3f' % value
        elif isinstance(value, np.ndarray):
            str_val = ''
            if all:
                str_val = 'np[' + ','.join(['%.2f' % val for val in value]) + ']'
            return str_val + 'mean{%.3f}' % np.mean(value)
        elif isinstance(value, torch.Tensor):
            value = value.to('cpu')
            # print(value, value.view(-1).shape, 'tensor value')
            if value.view(-1).shape[0] == 1:
                return '%.3f' % float(value)
            if all:
                str_val = 'torch[' + ','.join(['%.2f' % val for val in value]) + ']'
            return str_val + 'mean{%.3f}' % (value.mean())
        else:
            raise NotImplementedError(type(value), value)

    def _get_string_value(self, info, keyword: str, output_saved: bool, info_type: str):
        # str: keyword
        # iter, time, loss, error, etc.
        # info_type: loss/error
        assert info_type in self.info.keys(), 'info type should in self.info'
        tmp = []
        if keyword not in info.keys():
            return None
        for key, value in sorted(info.items()):
            if keyword in key:
                if '(error_count)' in key:  # do not print count
                    continue
                if info_type == 'error' and keyword == 'error':  # mean of this type
                    value = self.__error_div_count(info, key, value)
                nowstr = '%s:%s' % (key, self.__to_string(value, all=False))
                # print(key, keyword, keyword in key, div_name)
                if output_saved:
                    if info_type == 'error' and keyword == 'error':
                        value = self.info[info_type][key]
                        value = self.__error_div_count(self.info[info_type], key, value)
                        nowstr += '(mean=%s)' % self.__to_string(value)
                    else:
                        assert 'log_n_count' in self.info[info_type].keys(), 'output_saved(log_n_count) should in info[info_type]'
                        value = self.info[info_type][key]
                        divide = self.info[info_type]['log_n_count']
                        if divide != 1:
                            nowstr += '(mean=%s)' % self.__to_string(value / divide)
                if keyword == key:
                    ALL_VAL = nowstr
                else:
                    tmp.append(nowstr)
        assert len(tmp) != 0, '%s should have at least one type' % keyword
        return '{%s[%s]}' % (ALL_VAL, '|'.join(tmp))

    def _record_value(self, info, keywords, info_type):
        assert info_type in self.info.keys(), 'info type should in self.info'
        update_okay = False
        for key, value in info.items():
            for keyword in keywords:
                if keyword in key:
                    update_okay = True
                    # print('record key %s, infotype %s' % (key, info_type))
                    assert keyword in info.keys(), 'base name should in info.keys (%s)' % keyword
                    if key in self.info[info_type].keys():
                        self.info[info_type][key] += value
                    else:
                        self.info[info_type][key] = value
                    break
        if update_okay:
            if 'log_n_count' in self.info[info_type].keys():
                self.info[info_type]['log_n_count'] += 1
            else:
                self.info[info_type]['log_n_count'] = 1

    def _update_normal(self, info: dict, shouldprint: bool, pinfo):
        # if 'value' in info.keys():
        #     info.pop('value')
        # print(info, pinfo)
        self._direct_print(info, '_out')
        self._record_value(info, ['error', 'time', 'loss', 'n_count'], pinfo)
        if shouldprint:
            output = [pinfo]
            # iteration
            if 'iteration' in info.keys():
                nowstr = 'Iter: %d/%d(epoch%.2f)' % (info['iteration'][0], info['iteration'][1], info['iteration'][2])
                output.append(nowstr)
            if 'lr' in info.keys():
                nowstr = 'lr: %.10f' % info['lr']
                output.append(nowstr)
            output.append(self._get_string_value(info, 'time', True, pinfo))
            output.append(self._get_string_value(info, 'loss', True, 'loss'))
            output.append(self._get_string_value(info, 'error', True, 'error'))
            output = filter(lambda x: x is not None, output)
            self.logger.info(' '.join(output))

    def update_loss(self, info: dict, shouldprint: bool):
        self._update_normal(info, shouldprint, 'loss')
        if shouldprint:
            self.info['loss'].clear()
            # print('from loss: clean up')

    def update_error(self, info: dict, shouldprint: bool):
        self._update_normal(info, shouldprint, 'error')
        if info.get('flush', False):
            self.info['error'].clear()
            # print('from error: clean up')
