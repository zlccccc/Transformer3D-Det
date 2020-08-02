import torch.nn as nn

from .PointnetSG import PointNetSG

module_dict = {
    'pointnetSG': PointNetSG,
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
