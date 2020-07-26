import torch.nn as nn

from .PointnetPlus import PointnetPlusInitial
from .PointnetSSG import PointnetPlusSSGInitial
from .PointnetSSGSaveFeature import PointnetPlusSSGInitialSaveFeature
from .PointnetPlane import PointnetPlusPlane
from .PointnetPlaneRotate import PointnetPlusPlaneRotate
from .PointnetPointPlaneRotate import PointnetPlusPointPlaneRotate

module_dict = {
    'pointnetInitial': PointnetPlusInitial,
    'pointnetSSGInitial': PointnetPlusSSGInitial,
    'pointnetSSGInitialSaveFeature': PointnetPlusSSGInitialSaveFeature,
    'pointnetPlane': PointnetPlusPlane,
    'pointnetPlaneRotate': PointnetPlusPlaneRotate,
    'pointnetPointPlane': PointnetPlusPointPlaneRotate,
}


def model_entry(config):
    name = config.name
    if name not in module_dict.keys():
        return None
    del config['name']
    if config.get('use_syncbn', False):
        print('using SyncBatchNorm')
        raise NotImplementedError('SyncBatchNorm')
    print('get config from Pointnet Initial')
    print('module config:', config.keys())
    return module_dict[name](config)
