import torch.nn as nn

from .PointnetPlus import PointnetPlus
from .PointnetPlusSSG import PointnetPlusSSG

module_dict = {
    'pointnet++': PointnetPlus,
    'pointnet++2': PointnetPlusSSG,
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
