import importlib
import os
import re


def model_entry(config):
    print(__file__, __path__, '<<< file and path')
    for dirname in os.listdir(__path__[0]):
        packagepath = 'core.model.' + dirname
        print('try to load from ', dirname, packagepath)
        try:
            packagemodule = importlib.import_module(packagepath)
            func = packagemodule.model_entry
            model = func(config)
            if model is not None:
                return model
        except Exception as e:
            print('try loading: ', str(e))
            continue  # no model_entry
    raise NotImplementedError('Model Arch {} Not Implemented'.format(config.name))
