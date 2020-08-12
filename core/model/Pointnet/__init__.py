import torch.nn as nn

from .Pointnet import PointnetInit
from .PointnetPlus import PointnetPlus
from .PointnetPlusSSG import PointnetPlusSSG
from .PointnetPlusPartSeg import PointnetPlusPartSeg
from .PointnetPlusPartSegv2 import PointnetPlusPartSegv2

module_dict = {
    'pointnet': PointnetInit,
    'pointnet++': PointnetPlus,
    'pointnet++2': PointnetPlusSSG,
    'pointnet_partseg': PointnetPlusPartSeg,
    'pointnet_partsegv2': PointnetPlusPartSegv2,
}


def model_entry(config):
    name = config.name
    # print('model', module_dict.keys())
    if name not in module_dict.keys():
        return None
    del config['name']
    if config.get('use_syncbn', False):
        print('using SyncBatchNorm')
        raise NotImplementedError('SyncBatchNorm')
    print('get config from Pointnet MyImpilement')
    print('module config:', config.keys())
    return module_dict[name](config)
