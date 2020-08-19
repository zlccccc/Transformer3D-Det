import torch
import torch.nn as nn
from core.model.task_basemodel.taskmodel.seg_model import seg_module
from core.model.Pointnet.part_module.pointnet_utils import MLP
from .PointHRNetPartSeg import HighResolutionBlock
from core.model.PointnetYanx27 import provider
from core.model.task_error.ShapeNetError import ShapeNetError


class PointnetPlusPartSegHRv2(seg_module):
    def __init__(self, config):
        self.params = []
        self.task_type = config.get('task_type', 'No Impl')
        if self.task_type == 'ShapeNet':
            self.num_output = 50
            self.num_label = 16  # before fc (later use)
        else:
            raise NotImplementedError('task type %s' % self.task_type)

        normal_channel = config.get('normal_channel', True)
        self.config = config
        super(PointnetPlusPartSegHRv2, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        in_channel = in_channel + 3 + self.num_label  # initial xyz and mark
        # layer 0: 0->0,1; fc->fp and sa;
        self.layer0 = HighResolutionBlock([in_channel], 2)
        # layer 1: 0->0,1; 1->0,1,2  # nearby trans first
        self.layer1 = HighResolutionBlock(self.layer0.final_channels, 3)
        # layer 2: 0->0,1; 1->0,1,2; 2->1,2,3(3=feature)
        self.layer2 = HighResolutionBlock(self.layer1.final_channels, 4)  # TODO: CHANGE 3 TO 4
        # layer 3: 0->0,1; 1->0,1,2; 2->1,2; 3->2
        self.layer3 = HighResolutionBlock(self.layer2.final_channels, 3)
        # layer 4: 0->0,1; 1->0,1; 2->1
        self.layer4 = HighResolutionBlock(self.layer3.final_channels, 2)
        # layer 5: 0->0; 1->0; add initial channel
        self.layer5 = HighResolutionBlock(self.layer4.final_channels, 1)
        # layer 6: 0->0
        layer6_input = self.layer5.final_channels
        layer6_input[0] += in_channel
        self.layer6 = HighResolutionBlock(layer6_input, 1)
        self.fc = MLP(64, [self.num_output], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)
        self.init_relu = 'relu'
        self.init_params(nn.BatchNorm2d, init_type='kaiming_normal')

    def _forward(self, input):
        # print(xyz.shape)
        xyz = input['point_set']
        cls_label = input['cls']  # pointnet cls use

        B, N, _ = xyz.shape
        norm = xyz  # channel = 3 + (rgb)
        xyz = xyz[:,:,:3]

        cls_label = nn.functional.one_hot(cls_label, self.num_label).type_as(xyz)
        cls_label_one_hot = cls_label.view(B, 1, self.num_label).repeat(1, N, 1)
        xyz_feature = torch.cat([cls_label_one_hot, norm], dim=2)
        layer0_feature = [{'points': xyz, 'features': xyz_feature}]  # length
        layer1_feature = self.layer0(layer0_feature)
        layer2_feature = self.layer1(layer1_feature)
        layer3_feature = self.layer2(layer2_feature)
        layer4_feature = self.layer3(layer3_feature)
        layer5_feature = self.layer4(layer4_feature)
        layer6_feature = self.layer5(layer5_feature)
        layer6_feature[0]['features'] = torch.cat([layer6_feature[0]['features'], xyz_feature], dim=-1)
        # print('shape', layer6_feature[0]['features'].shape)
        layer7_feature = self.layer6(layer6_feature)
        x = layer7_feature[0]['features']
        x = self.fc(x)
        # print(x.shape, ' <<< result xyz and feature shape')
        return {'value': x}

    def _before_forward(self, input):
        # print(input['point_set'].shape, 'before forward; TODO CHECK IT')
        if self.mode == 'train':
            points = input['point_set'].cpu().data.numpy()
            # points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.from_numpy(points)
            input['point_set'] = points.type_as(input['point_set'])
            # print('before forwrad')
        return input

    def calculate_error(self, input, output):
        output = super(PointnetPlusPartSegHRv2, self).calculate_error(input, output)
        if self.task_type == 'ShapeNet':
            output = ShapeNetError(input, output)
        return output

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
    net = PointnetPlusPartSegHRv2(config)
    net = net.cuda()
    net.set_params()
    # exit()
    from torchsummary import summary

    summary(net, batch_size=2, input_size=(4096, 3))
