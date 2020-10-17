import importlib
import os
import re
import ipdb

def model_entry(config):
    print(__file__, __path__, '<<< file and path')
    for dirname in os.listdir(__path__[0]):
        if '__pycache__' in dirname or '.' in dirname:  # cache or file
            continue
        packagepath = 'core.model.' + dirname
        # print('try to load from ', dirname, packagepath)
        try:
            # ipdb.set_trace()
            packagemodule = importlib.import_module(packagepath)
            func = packagemodule.model_entry
            model = func(config)
            if model is not None:
                return model
        except Exception as e:
            exception_type = str(e)
            if 'has no attribute \'model_entry\'' in exception_type:
                print('try loading: ', exception_type)  # no model_entry
                continue
            raise e
    raise NotImplementedError('Model Arch {} Not Implemented'.format(config.name))
