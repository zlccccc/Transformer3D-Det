from .dataset.Test3DDataset import Test3DDataset
from .dataset.Classification3D.ModelNetDataset import ModelNetDataset
from .dataset.Classification3D.ModelNetFeatureDataset import ModelNetFeatureDataset
from .dataset.Language3D._3DSSGDataset import PCSGDataset
from .dataset.Segmentation3D.ShapeNetDataset import ShapeNetDataset
from .dataset.SementicDataset.Semantic3D import Semantic3DDataset


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
        return PCSGDataset(**config)
    elif name == 'ShapeNetDataset':
        return ShapeNetDataset(**config)
    elif name == 'Semantic3DDataset':
        return Semantic3DDataset(**config)
    else:
        raise NotImplementedError(name)


def get_dataset(config: dict):
    output = {}
    for key, value in config.items():
        output[key] = get_one_dataset(value)
    return output
