import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import pytorch_utils as pt_utils
from .utils.helper_tool import ConfigSemanticKITTI, ConfigS3DIS, ConfigSemantic3D
from .utils.helper_tool import DataProcessing as DP
from .utils.OutputUtils import Semantic3DModelTester
import numpy as np
from sklearn.metrics import confusion_matrix
from .RandLANet import RandLANet, reduce_points, compute_loss, IoUCalculator, compute_acc
from core.model.task_basemodel.backbone.base_model import base_module
from thop import profile, clever_format


class RandLANetv1(base_module):
    def __init__(self, config):
        super(RandLANetv1, self).__init__()
        self.backbone = RandLANet(config)
        dataset_name = config.get('dataset', 'SemanticKITTI')  # ONLY GET NAME
        self.dataset_name = dataset_name
        self.calculator, self.iou_n_count = None, 1
        self.thop_cal = False  # True
        # self.thop_cal = True
        if dataset_name == 'Semantic3D':
            self.config = ConfigSemantic3D
        elif dataset_name == 'SemanticKITTI':
            self.config = ConfigSemanticKITTI
        else:
            raise NotImplementedError(dataset_name)

    def _forward(self, input):
        if self.thop_cal:
            flops, params = profile(self.backbone, inputs=(input, ))
            print(clever_format([flops, params], '%.3f'), flush=True)
            exit()
        return self.backbone(input)

    def calculate_loss(self, input, output):
        # print(self.mode, flush=True)
        output['labels'] = input['labels']
        output = reduce_points(output, self.config)
        loss, output = compute_loss(output, self.config)
        acc, output = compute_acc(output)
        output['acc(loss)'] = output['acc']
        return output

    def calculate_error(self, input, output):
        # print(self.mode, flush=True)
        output['labels'] = input['labels']
        output = reduce_points(output, self.config)
        acc, output = compute_acc(output)
        output['acc(error)'] = output['acc']
        output['acc(error_count)'] = 1
        output['error'] = 1 - output['acc(error)']
        if self.calculator is not None:  # already last
            self.calculator.add_data(output)
            mean_iou, iou_list = self.calculator.compute_iou()
            iou_list = np.array(iou_list)
            # print(mean_iou, iou_list)
            output['mean_iou(error)(mean)'] = mean_iou
            output['iou_list(error)(mean)'] = iou_list
            self.iou_n_count = 0
        output['n_count'] = 1
        return output

    def initialize_error(self):  # TODO; IoUCalculator; must for dataset
        print('---- Initialize error calculating ----')
        self.calculator = IoUCalculator(self.config)
        self.iou_n_count = 1

    def save_dataset(self, dataset):
        if self.dataset_name == 'Semantic3D':
            tester = Semantic3DModelTester(self, dataset)
        else:
            raise NotImplementedError(self.dataset_name)
        tester.test()
