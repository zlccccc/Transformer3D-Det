from .iterRunner import iterRunner


def getrunner(config):
    name = config.name
    print('using runner %s' % name)
    if name == 'iteration':
        return iterRunner
    else:
        raise NotImplementedError(name)