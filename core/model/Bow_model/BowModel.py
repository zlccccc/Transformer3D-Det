import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model.task_basemodel.taskmodel.cls_model import cls_module
from core.runner.runner_utils.bow_util import initialize_centers, compute_centers


class BowModel(cls_module):
    def __init__(self, config):
        super(BowModel, self).__init__()
        self.params = []
        num_channel = config.num_channel
        num_centers = config.num_centers
        num_class = config.num_output
        self.bow_dict = nn.Parameter(torch.FloatTensor(num_centers, num_channel).uniform_(-1, 1))
        self.register_parameter('BOW_Weight', self.bow_dict)
        base_channels = 128
        self.fc1 = nn.Linear(num_centers, base_channels * 4)
        self.fc2 = nn.Linear(base_channels * 4, base_channels * 4)
        self.fc3 = nn.Linear(base_channels * 4, base_channels * 4)
        self.bn1 = nn.BatchNorm1d(base_channels * 4)
        self.bn2 = nn.BatchNorm1d(base_channels * 4)
        self.bn3 = nn.BatchNorm1d(base_channels * 4)
        self.cls = nn.Linear(4 * base_channels, num_class)
        self.init_relu = 'relu'
        self.init_params(nn.BatchNorm2d, init_type='kaiming_normal')

    def _record_bow_dict(self, bow_dict):
        assert bow_dict.shape == self.bow_dict.shape, 'shape should be same'
        # print(self.bow_dict.data)
        self.bow_dict.data = bow_dict

    def _forward(self, input):
        x = input['point_set']
        x, SC, CC = compute_centers(x, self.bow_dict)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        result = self.cls(x)
        return {'value': result}


if __name__ == "__main__":
    import sys
    import os
    from easydict import EasyDict

    config = {
        'num_output': 83,
        'normal_channel': False
    }
    config = EasyDict(config)
    print(os.getcwd())
    net = BowModel(config)
    net = net.cuda()
    net.set_params()
    # exit()
    from torchsummary import summary

    summary(net, batch_size=2, input_size=(4096, 3))
