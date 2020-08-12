from tensorboardX import SummaryWriter


class logger():
    def __init__(self, path):
        self.logger = SummaryWriter(path + '/events')

    def update(self, info: dict, floatvalue: dict, shouldprint: bool, type: str):  # type: loss/error
        if type == 'error' and not info.get('flush', False):  # should not print
            return
        if type == 'error' and 'last_iter' not in info.keys():  # should not print
            return
        for key, value in floatvalue.items():
            if type == 'loss':
                iter = info['iteration'][0]
            elif type == 'error':
                iter = info['last_iter']
            else:
                raise NotImplementedError(type)
            # print(iter, '<< add scalar iter')
            self.logger.add_scalar(key, value, iter)
