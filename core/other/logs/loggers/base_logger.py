from core.utils.utils import create_logger
from .loggerRegistry import LOGGERS


@LOGGERS.register_module()
class baselogger():
    def __init__(self, path):
        self.logger = create_logger('base', path)
        self.info = {'loss': {}, 'error': {}}

    def _direct_print(self, info, keyword: str):
        for key, value in info.items():
            if keyword in key:
                self.logger.info("%s:%s" % (key, str(value)))

    def _get_string_value(self, info, keyword: str, output_saved: bool, div_name: str, info_type: str):
        # str: keyword
        # iter, time, loss, error, etc.
        # info_type: loss/error
        assert info_type in self.info.keys(), 'info type should in self.info'
        tmp = []
        if keyword not in info.keys():
            return None
        for key, value in info.items():
            if keyword in key:
                if info_type == 'error' and keyword == 'error' and 'n_count' in info.keys():
                    value = float(value) / info['n_count']  # mean of this iteration
                nowstr = '%s:%.3f' % (key, value)
                # print(key, keyword, keyword in key, div_name)
                if output_saved:
                    assert div_name in self.info[info_type].keys(), 'output_saved(%s) should in info[info_type]' % div_name
                    if self.info[info_type][div_name] != 1:
                        nowstr += '(mean=%.3f)' % (self.info[info_type][key] / self.info[info_type][div_name])
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
                        self.info[info_type][key] += float(value)
                    else:
                        self.info[info_type][key] = float(value)
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
            output.append(self._get_string_value(info, 'time', True, 'log_n_count', pinfo))
            output.append(self._get_string_value(info, 'loss', True, 'log_n_count', 'loss'))
            output.append(self._get_string_value(info, 'error', True, 'n_count', 'error'))
            output = filter(lambda x: x is not None, output)
            self.logger.info(' '.join(output))

    def update_loss(self, info: dict, shouldprint: bool):
        self._update_normal(info, shouldprint, 'loss')
        if shouldprint:
            self.info['loss'].clear()
            #print('from loss: clean up')

    def update_error(self, info: dict, shouldprint: bool):
        self._update_normal(info, shouldprint, 'error')
        if info.get('flush', False):
            self.info['error'].clear()
            #print('from error: clean up')
