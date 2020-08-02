import torch.nn as nn

from .Bow_model import model_entry as BowModel
from .Pointnet import model_entry as Pointnet
from .PointnetYanx27 import model_entry as PointnetInitial
from .SGmodel import model_entry as SGmodel


model_entry_name = [
    Pointnet,
    PointnetInitial,
    BowModel,
    SGmodel,
]


def model_entry(config):
    for func in model_entry_name:
        model = func(config)
        if model is not None:
            return model
    raise NotImplementedError('Model Arch {} Not Implemented'.format(config.name))
