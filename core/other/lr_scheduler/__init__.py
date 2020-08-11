from .schedular import StepLRScheduler, CosineLRScheduler
from torch.optim.lr_scheduler import StepLR


def get_lr_scheduler(config):
    name = config.type
    del config['type']
    if name == 'STEP':
        return StepLRScheduler(**config)
    elif name == 'COSINE':
        return CosineLRScheduler(**config)
    elif name == 'lr_scheduler_step':
        print(config.keys(), '  <<< using torch-StepLR')
        config.pop('base_lr')  # base_lr only for print (get lr not use)
        config['last_epoch'] = config['last_iter']
        config.pop('last_iter')
        return StepLR(**config)
    else:
        raise RuntimeError('unknown lr_scheduler type: {}'.format(name))
