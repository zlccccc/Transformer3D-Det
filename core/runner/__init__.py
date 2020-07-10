from .iterRunner import iterRunner
from .testRunner import testRunner


def getrunner(config):
    name = config.name
    print('using runner %s' % name)
    if name == 'iteration':
        return iterRunner
    elif name == 'test':
        return testRunner
    else:
        raise NotImplementedError(name)