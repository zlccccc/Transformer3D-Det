import torch.nn as nn

from .Pointnet.PointnetPlus import PointnetPlus

def model_entry(config):
    name = config.name
    del config['name']
    BN = nn.BatchNorm1d
    if config.get('use_syncbn', False):
        print('using SyncBatchNorm')
        raise NotImplementedError('SyncBatchNorm')
    if name == 'pointnet++':
        print(config.keys())
        return PointnetPlus(config)

    raise NotImplementedError('Model Arch {} Not Implemented'.format(name))
