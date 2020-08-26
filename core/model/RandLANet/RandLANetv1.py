import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import pytorch_utils as pt_utils
from .utils.helper_tool import ConfigSemanticKITTI, ConfigS3DIS, ConfigSemantic3D
from .utils.helper_tool import DataProcessing as DP
import numpy as np
from sklearn.metrics import confusion_matrix
from .RandLANet import RandLANet, compute_loss, IoUCalculator, compute_acc
from core.model.task_basemodel.backbone.base_model import base_module


class RandLANetv1(base_module):
    def __init__(self, config):
        super(RandLANetv1, self).__init__()
        self.backbone = RandLANet(config)
        dataset_name = config.get('dataset', 'SemanticKITTI')  # ONLY GET NAME
        if dataset_name == 'Semantic3D':
            self.config = ConfigSemantic3D
        else:
            raise NotImplementedError(dataset_name)

    def _forward(self, input):
        return self.backbone(input)

    def calculate_loss(self, input, output):
        output['labels'] = input['labels']
        loss, output = compute_loss(output, self.config)
        acc, output = compute_acc(output)
        output['acc(loss)'] = output['acc']
        return output

    def calculate_error(self, input, output):
        output['labels'] = input['labels']
        loss, output = compute_loss(output, self.config)
        del output['loss']
        acc, output = compute_acc(output)
        output['acc(error)'] = output['acc']
        output['acc(error_count)'] = 1
        output['error'] = 1 - output['acc(error)']
        output['n_count'] = 1
        return output

    def calculate_dataset_initialize(self):  # TODO; IoUCalculator; must for dataset
        self.calculator = IoUCalculator(self.config)
