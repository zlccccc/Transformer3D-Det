import torch.optim as optim


def get_optimizer(config, parameters):
    name = config.name
    del config['name']
    print('using optimizer %s' % name)
    print(config)
    if name == "Adam":
        return optim.Adam(params=parameters, **config)
    elif name == 'SGD':
        return optim.SGD(params=parameters, **config)
    elif name == "AdamW":
        return optim.AdamW(params=parameters, **config)
    elif name == 'myAdamW':
        from .AdamW import AdamW
        return AdamW(params=parameters, **config)
    else:
        raise NotImplementedError(name)
