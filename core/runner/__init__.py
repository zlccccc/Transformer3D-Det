from .iterRunner import iterRunner
from .testRunner import testRunner
from .bowRunner import bowRunner


def getrunner(config):
    name = config.name
    print('using runner %s' % name)
    if name == 'iteration':
        return iterRunner
    elif name == 'bow':
        return bowRunner
    elif name == 'test':
        return testRunner
    else:
        raise NotImplementedError(name)
