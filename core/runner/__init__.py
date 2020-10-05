from .iterRunner import iterRunner
from .iterBNDecayRunner import iterBNDecayRunner
from .testRunner import testRunner
from .bowRunner import bowRunner
from .saveRunner import saveRunner


def getrunner(config):
    name = config.name
    print('using runner %s' % name)
    if name == 'iteration':
        return iterRunner
    if name == 'iterBNDecay':
        return iterBNDecayRunner
    elif name == 'bow':
        return bowRunner
    elif name == 'test':
        return testRunner
    elif name == 'save':
        return saveRunner
    else:
        raise NotImplementedError(name)
