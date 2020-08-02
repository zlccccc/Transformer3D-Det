from .dataset.Test3DDataset import Test3DDataset
from .dataset.ModelNetDataset import ModelNetDataset
from .dataset.ModelNetFeatureDataset import ModelNetFeatureDataset
from .dataset._3DSSGDataset import PCSGDataset


def get_one_dataset(config: dict):
    name = config.name
    del config['name']
    print(config)
    if name == 'Test3DDataset':
        return Test3DDataset(**config)
    elif name == 'ModelNetDataset':
        return ModelNetDataset(**config)
    elif name == 'ModelNetFeatureDataset':
        return ModelNetFeatureDataset(**config)
    elif name == '3DSSGDataset':
        return ModelNetFeatureDataset(**config)
    else:
        raise NotImplementedError(name)


def get_dataset(config: dict):
    output = {}
    for key, value in config.items():
        output[key] = get_one_dataset(value)
    return output
