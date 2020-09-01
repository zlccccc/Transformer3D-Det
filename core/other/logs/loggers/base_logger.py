from core.utils.utils import create_logger
from colorama import Fore, Style
import numpy as np
import torch


class logger():
    def __init__(self, path):
        self.logger = create_logger('base', path)
        self.clean_log_file = path + '.clean'
        f = open(self.clean_log_file, 'w')
        f.close()
        self.output = {}
        self.info = {'loss': {}, 'error': {}}
        self.final_value = {}  # key; float(value)

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
            # print(value,divide)
            value = value / divide  # mean of this iteration
        else:
            raise NotImplementedError('Error fac name not in keyword; error_key = %s' % key)
        return value

    def __to_string(self, value, all=True, ForeColor=Fore.LIGHTWHITE_EX):
        if isinstance(value, (float, int)):
            floatvalue = float(value)
            return '%s%.3f%s' % (ForeColor, floatvalue, Style.RESET_ALL), floatvalue
        elif isinstance(value, np.ndarray):
            str_val = '' if not all else 'np[' + ','.join(['%.2f' % val for val in value]) + ']'
            floatvalue = float(np.mean(value))
            return str_val + 'arrmean{%s%.3f%s}' % (ForeColor, floatvalue, Style.RESET_ALL), floatvalue
        elif isinstance(value, torch.Tensor):
            value = value.to('cpu')
            # print(value, value.view(-1).shape, 'tensor value')
            if value.view(-1).shape[0] == 1:
                floatvalue = float(value)
                return '%s%.3f%s' % (ForeColor, floatvalue, Style.RESET_ALL), floatvalue
            # print(value, value.view(-1).shape)
            str_val = '' if not all else 'torch[' + ','.join(['%.2f' % val for val in value]) + ']'
            # print('log', value) TODO
            # print('forward ?? err? ', value)
            floatvalue = float(torch.mean(value))
            # print(floatvalue, str_val)
            return str_val + 'arrmean{%s%.3f%s}' % (ForeColor, floatvalue, Style.RESET_ALL), floatvalue
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
            if keyword in key:  # get one output
                if '(error_count)' in key:  # do not print count
                    continue
                if info_type == 'error' and keyword == 'error':  # mean of this type
                    value = self.__error_div_count(info, key, value)
                dir_output_all = False if '(mean)' not in key else True
                strval, _ = self.__to_string(value, all=dir_output_all, ForeColor=Fore.LIGHTBLUE_EX)
                nowstr = '%s:%s' % (key, strval)
                if dir_output_all:
                    self.final_value[key] = _
                # floatvalue not useful
                # print(key, keyword, keyword in key, div_name)
                if output_saved and not dir_output_all:
                    if info_type == 'error' and keyword == 'error':
                        value = self.info[info_type][key]
                        value = self.__error_div_count(self.info[info_type], key, value)
                        strval, floatval = self.__to_string(value, ForeColor=Fore.LIGHTCYAN_EX)
                        self.final_value[key] = floatval
                        nowstr += '(mean=%s)' % strval
                    else:
                        assert 'log_n_count' in self.info[info_type].keys(), 'output_saved(log_n_count) should in info[info_type]'
                        value = self.info[info_type][key]
                        divide = self.info[info_type]['log_n_count']
                        if keyword == 'time':
                            strval, floatval = self.__to_string(value / divide, ForeColor=Fore.LIGHTRED_EX)
                            # self.final_value[key] = floatval  # do not log time at tensorboard
                        elif keyword == 'loss':
                            strval, floatval = self.__to_string(value / divide, ForeColor=Fore.LIGHTGREEN_EX)
                            self.final_value[key] = floatval
                        else:
                            raise NotImplementedError('when calculating mean and coloring', keyword)
                        if divide != 1:
                            nowstr += '(mean=%s)' % strval
                if keyword == key:
                    ALL_VAL = nowstr
                else:
                    tmp.append(nowstr)
        assert len(tmp) != 0, '%s should have at least one type' % keyword
        ALL_VAL = '%s%s%s' % (Fore.LIGHTYELLOW_EX, ALL_VAL, Style.RESET_ALL)
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
        # baseline
        # print(info, pinfo)
        self._direct_print(info, '_out')
        self._record_value(info, ['error', 'time', 'loss', 'n_count'], pinfo)
        if shouldprint:
            output = [pinfo]
            # iteration
            if 'iteration' in info.keys():
                epochstr = 'epoch%.2f' % info['iteration'][2]
                epochstr = '%s%s%s' % (Fore.RED, epochstr, Style.RESET_ALL)
                iterstr = '%d/%d' % (info['iteration'][0], info['iteration'][1])
                iterstr = '%s%s%s' % (Fore.YELLOW, iterstr, Style.RESET_ALL)
                nowstr = 'Iter: %s(%s)' % (iterstr, epochstr)
                output.append(nowstr)
            if 'lr' in info.keys():
                nowstr = 'lr: %.10f' % info['lr']
                nowstr = '%s%s%s' % (Fore.LIGHTWHITE_EX, nowstr, Style.RESET_ALL)
                self.final_value['lr'] = float(info['lr'])
                output.append(nowstr)
            output.append(self._get_string_value(info, 'time', True, pinfo))
            output.append(self._get_string_value(info, 'loss', True, 'loss'))
            output.append(self._get_string_value(info, 'error', True, 'error'))
            output = filter(lambda x: x is not None, output)
            value = ' '.join(output)
            self.logger.info(value)
            clean_log_file = open(self.clean_log_file, 'a')
            for col in vars(Fore).values():
                value = value.replace(str(col), '')
            value = value.replace(str(Style.RESET_ALL), '')
            clean_log_file.write(value+'\n')
            clean_log_file.close()

    def update_loss(self, info: dict, shouldprint: bool):
        self.final_value = {}
        self._update_normal(info, shouldprint, 'loss')
        if shouldprint:
            self.info['loss'].clear()
            # print('from loss: clean up')

    def update_error(self, info: dict, shouldprint: bool):
        self.final_value = {}
        self._update_normal(info, shouldprint, 'error')
        if info.get('flush', False):
            self.info['error'].clear()
            # print('from error: clean up')

    def get_float_value(self):
        return self.final_value
