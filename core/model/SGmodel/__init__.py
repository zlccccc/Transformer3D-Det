import torch.nn as nn
import importlib
import os

module_dict = {
    'pointnetSG': 'PointNetSG.PointNetSG',
}


def model_entry(config):
    package_name = __file__
    print('package name', package_name)
    package_name = package_name.split('\\')[-2]
    name = config.name
    print('try loading from model package %s: ' % package_name, module_dict.keys())
    if name not in module_dict.keys():
        return None
    del config['name']
    if config.get('use_syncbn', False):
        print('using SyncBatchNorm')
        raise NotImplementedError('SyncBatchNorm')
    print('get config from %s MyImpilement' % package_name)
    print('module config:', config.keys())
    try:
        pathlist = 'core.model.' + package_name + '.' + module_dict[name]  # must abspath
        pathlist = pathlist.split('.')
        relativepath, packagepath = pathlist[-1], '.'.join(pathlist[:-1])
        print('try to import_module', relativepath, packagepath)
        package = importlib.import_module(packagepath)
        assert hasattr(package, relativepath), 'should have class in python file'
        modelclass = getattr(package, relativepath)
    except Exception as e:
        print('dataset path', pathlist, 'not exist')
        print(str(e))
        modelclass = None
    return modelclass(config)
