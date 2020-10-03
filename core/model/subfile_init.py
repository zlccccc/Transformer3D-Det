import importlib
import os
import re


def sub_model_entry(config, package_name, module_dict):
    package_name = re.split(r'[\\/]', package_name)
    # print('SPLITED package name', package_name)
    package_name = package_name[-2]
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
        print('model path', pathlist, 'not exist')
        print(str(e))
        modelclass = None
        raise NotImplementedError(pathlist)
    return modelclass(config)
