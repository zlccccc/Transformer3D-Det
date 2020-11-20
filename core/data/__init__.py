# from .dataset.Test3DDataset import Test3DDataset
# from .dataset.Classification3D.ModelNetDataset import ModelNetDataset
# from .dataset.Classification3D.ModelNetFeatureDataset import ModelNetFeatureDataset
# from .dataset.Language3D._3DSSGDataset import PCSGDataset
# from .dataset.Segmentation3D.ShapeNetDataset import ShapeNetDataset
# from .dataset.SementicDataset.Semantic3D import Semantic3DDataset
# from .dataset.SementicDataset.SemanticKITTI import SemanticKITTIDataset
import importlib

dataset_dict = {
    'Test3DDataset': 'Test3DDataset.Test3DDataset',
    'ModelNetDataset': 'Classification3D.ModelNetDataset.ModelNetDataset',
    'ModelNetFeatureDataset': 'Classification3D.ModelNetFeatureDataset.ModelNetFeatureDataset',
    '3DSSGDataset': 'Language3D._3DSSGDataset.PCSGDataset',
    'ShapeNetDataset': 'Segmentation3D.ShapeNetDataset.ShapeNetDataset',
    'Semantic3DDataset': 'SementicDataset.Semantic3D.Semantic3DDataset',
    'SemanticKITTIDataset': 'SementicDataset.SemanticKITTI.SemanticKITTIDataset',
    'SemanticKITTIAUGDataset': 'SementicDataset.SemanticKITTIAUG.SemanticKITTIDataset',
    'Scannetv2': 'Detection3D.scannet.scannet_detection_dataset.ScannetDetectionDataset',
    'Sunrgbd': 'Detection3D.sunrgbd.sunrgbd_detection_dataset.SunrgbdDetectionVotesDataset',
    # 'Scanrefer': 'Detection3D.scanrefer.dataset.ScannetReferenceDataset',
    'Scanrefer': 'Detection3D.scanrefer.dataset.GetScanReferDataset',
}


def get_one_dataset(config: dict):
    name = config.name
    del config['name']
    print('support dataset', dataset_dict.keys())
    print('trying to find dataset', config)
    if name not in dataset_dict.keys():
        raise NotImplementedError(name)
    try:
        pathlist = 'core.data.dataset.' + dataset_dict[name]  # must abspath
        pathlist = pathlist.split('.')
        relativepath, packagepath = pathlist[-1], '.'.join(pathlist[:-1])
        print('Try to import_module', relativepath, 'from path', packagepath)
        package = importlib.import_module(packagepath)
        assert hasattr(package, relativepath), 'should have class in python file'
        datasetclass = getattr(package, relativepath)
        print('Load model path success!')
    except Exception as e:
        print('dataset path', pathlist, 'not exist')
        print(str(e))
        raise e
        datasetclass = None
    return datasetclass(**config)


def get_dataset(config: dict):
    output = {}
    for key, value in config.items():
        output[key] = get_one_dataset(value)
    return output
