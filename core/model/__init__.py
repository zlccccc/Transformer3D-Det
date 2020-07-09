import torch.nn as nn

from .Pointnet.PointnetPlus import PointnetPlus
from .PointnetYanx27.PointnetPlus import PointnetPlusInitial
from .PointnetYanx27.PointnetSSG import PointnetPlusSSGInitial


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
    if name == 'pointnetSSGInitial':
        print(config.keys())
        return PointnetPlusSSGInitial(config)
    raise NotImplementedError('Model Arch {} Not Implemented'.format(name))
