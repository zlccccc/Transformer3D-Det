import torch.optim as optim
from .schedular import *


def get_lr_scheduler(config):
    name = config.type
    del config['type']
    if name == 'STEP':
        return StepLRScheduler(**config)
    elif name == 'COSINE':
        return CosineLRScheduler(**config)
    else:
        raise RuntimeError('unknown lr_scheduler type: {}'.format(name))
