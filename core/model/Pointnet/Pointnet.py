import torch.nn as nn
from core.model.task_basemodel.taskmodel.cls_model import cls_module
from core.model.Pointnet.part_module.pointnet_utils import PointNetMSG, PointNetFeature


class PointnetInit(cls_module):
    def __init__(self, config):
        self.params = []
        assert 'normal_channel' not in config.keys()  # not same as PointnetInitial
        num_output = config.get('num_output', 100)
        self.in_channel = config.get('in_channel', 0)  # more
        # self.base_lr = config.base_lr
        # self.weight_decay = config.weight_decay
        self.config = config
        super(PointnetInit, self).__init__()
        self.fc1 = PointNetFeature(self.in_channel, [64, 128, 1024], [512, 256, num_output], [0, 0])
        self.init_relu = 'relu'
        self.init_params(nn.BatchNorm2d, init_type='kaiming_normal')

    def _forward(self, input):
        # print(xyz.shape)
        xyz = input['point_set']

        B, _, _ = xyz.shape
        if self.in_channel:
            norm = xyz[:, :, 3:]
        else:
            norm = None
        xyz = xyz[:, :, :3]
        x = self.fc1(xyz, norm)
        return {'value': x}


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
    net = PointnetPlusSSG(config)
    net = net.cuda()
    net.set_params()
    # exit()
    from torchsummary import summary

    summary(net, batch_size=2, input_size=(4096, 3))
