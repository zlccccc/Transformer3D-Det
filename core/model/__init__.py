import torch.nn as nn

from .Pointnet.PointnetPlus import PointnetPlus
from .PointnetYanx27.PointnetPlus import PointnetPlusInitial


def model_entry(config):
    name = config.name
    del config['name']
    if config.get('use_syncbn', False):
        print('using SyncBatchNorm')
        raise NotImplementedError('SyncBatchNorm')
    if name == 'pointnet++':
        print(config.keys())
        return PointnetPlus(config)
    if name == 'pointnetInitial':
        print(config.keys())
        return PointnetPlusInitial(config)

    raise NotImplementedError('Model Arch {} Not Implemented'.format(name))
