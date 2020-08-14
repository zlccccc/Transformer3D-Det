import torch
import torch.nn as nn
from core.model.task_basemodel.taskmodel.seg_model import seg_module
from core.model.Pointnet.part_module.pointnet_utils import PointNetMSG, PointNetFeature, PointNetPropagation, MLP
from core.model.PointnetYanx27 import provider
from core.model.task_error.ShapeNetError import ShapeNetError


class HighResolutionBlock(nn.Module):
    def __init__(self, in_channels, out_length, way='fusion'):
        assert abs(len(in_channels) - out_length) <= 1
        self.moduledict = nn.ModuleList()
        final_channels = [0 for i in range(out_length)]
        fc_channels = [64, 128, 256]
        for id, channel in enumerate(in_channels):
            module_name = "%d" % id
            if len(id) < len(fc_channels):
                raise NotImplementedError('HRNet: Self-Extension Not Implemented')
            _in, fin = in_channels[id], fc_channels[id]
            nowmodel = MLP(_in, [fin, fin], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)  # self64+down128

        for id, channel in enumerate(in_channels):
            for out_id in range(-1, out_length):
                module_name = "_%d%d" % (id, out_id)
                _in = fc_channels[id]  # input channel(after fc)
                if id == out_id or out_id:
                    out_channels = [64, 128, 256]
                    if id < len(out_channel):
                        fin = out_channels[id]
                        nowmodel = MLP(in_channels, [fin, fin], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)  # self64+down128
                        out_channels[out_id] += fin
                    else:
                        raise NotImplementedError('HRNet: Self-Extension Not Implemented')
                elif id + 1 == out_id:
                    if id == 0:
                        nowmodel, fin = PointNetMSG(512, [0.2], [32], _in, [[64, 128]], [128]), 128
                    elif id == 1:
                        nowmodel, fin = PointNetMSG(128, [0.4], [64], _in, [[128, 256]], [256]), 256
                    elif id == 2:
                        nowmodel, fin = PointNetFeature(_in, [256, 512, 1024], [1024]), 1024
                    else:
                        raise NotImplementedError('HRNet: downsample-%s Not Implemented', % module_name)
                elif id - 1 == out_id:
                    if id == 1 or id == 2 or id == 3:
                        nowmodel, fin = PointNetPropagation(in_channel=_in, mlp=_in), _in
                    else:
                        raise NotImplementedError('fp ???')
                else:
                    continue
                self.moduledict[module_name] = nowmodel
                self.final_channels[out_id] += fin


class PointnetPlusPartSegHR(seg_module):
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
        super(PointnetPlusPartSegHR, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        in_channel = in_channel + 3 + self.num_label  # initial xyz and mark
        # layer 0: 0->0,1; fc->fp and sa;
        self.fc0_0 = MLP(in_channel, [64, 64], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)
        self.sa0_01 = PointNetMSG(512, [0.2], [32], 64, [[64, 128]], [128])  # downsample; record(have related feature delxyz)
        # feature: 0:(2048, 64); 1: (512, 128); 2: (128, 256)
        # layer 1: 0->0,1; 1->0,1,2  # nearby trans first
        self.fc1_0 = MLP(64, [64, 64], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)
        self.fc1_1 = MLP(128, [128, 128], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)
        self.fp1_10 = PointNetPropagation(in_channel=128, mlp=[128])
        self.sa1_01 = PointNetMSG(512, [0.2], [32], 64, [[64, 128]], [128])
        self.sa1_12 = PointNetMSG(128, [0.4], [64], 128, [[128, 256]], [256])
        # layer 2: 0->0,1; 1->0,1,2; 2->1,2,3(3=feature); 3->2
        self.fc2_0 = MLP(64 + 128, [64, 64], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)  # self64+down128
        self.fc2_1 = MLP(128 + 128, [128, 128], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)
        self.fc2_2 = MLP(256, [256, 256], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)
        self.fp2_10 = PointNetPropagation(in_channel=128, mlp=[128])
        self.fp2_21 = PointNetPropagation(in_channel=256, mlp=[256])
        self.sa2_01 = PointNetMSG(512, [0.2], [32], 64, [[64, 128]], [128])
        self.sa2_12 = PointNetMSG(128, [0.4], [64], 128, [[128, 256]], [256])
        self.sa2_23 = PointNetFeature(64 + 128, [256, 512, 1024], [1024])  # in; mlp; fc
        # layer 3: 0->0,1; 1->0,1,2; 2->1,2; 3->2
        self.fc2_0 = MLP(64 + 128, [64, 64], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)  # self64+down128
        self.fc2_1 = MLP(128 + 128, [128, 128], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)
        self.fc2_2 = MLP(256 + 256, [256, 256], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)
        self.fc2_3 = MLP(1024, [], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)  # do not change it
        self.fp2_10 = PointNetPropagation(in_channel=128, mlp=[128])
        self.fp2_21 = PointNetPropagation(in_channel=256, mlp=[256])
        self.sa2_01 = PointNetMSG(512, [0.2], [32], 64, [[64, 128]], [128])
        self.sa2_12 = PointNetMSG(128, [0.4], [64], 128, [[128, 256]], [256])
        # layer 4



        # self.sa1 = PointNetMSG(512, [0.2], [32], 3 + in_channel, [[64, 64, 128]], [])  # should input xyz...
        # self.sa2 = PointNetMSG(128, [0.4], [64], 128, [[128, 128, 256]], [])
        # self.fc1 = PointNetFeature(256, [256, 512, 1024], [])  # in; mlp; fc
        # self.fp3 = PointNetPropagation(in_channel=1280, mlp=[256, 256])
        # self.fp2 = PointNetPropagation(in_channel=384, mlp=[256, 128])
        # self.fp1 = PointNetPropagation(in_channel=128 + 3 + in_channel + self.num_label, mlp=[128, 128])
        # self.mlp1 = MLP_List(128, [128, self.num_output], dropout=[0.5], FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=None)

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
        # group: for downsample use
        # layer: 1-0,1 (0->1)
        # layer 0: 0->0,1; fc->fp and sa;
        l00_feature = self.fc0_0(l00_feature)
        l1_xyz, l01__feature, l1_group = self.fc1_0(xyz, l00_feature, return_group_id=True)
        l1_xyz, l11_feature, l1_group = self.sa0_1(xyz, l00_feature, return_group_id=True)
        l2_xyz, l22_feature, l2_group = self.sa2(l1_xyz, l11_feature, return_group_id=True)
        # layer: 3-0,1,2,3
        l33_feature = self.fc1(l2_xyz, l22_feature)
        l33_feature = l33_feature.view(l33_feature.shape[0], 1, l33_feature.shape[1])
        l3_xyz = torch.zeros((l2_xyz.shape[0], 1, 3)).type_as(xyz)
        # l1_xyz; l2_xyz; l3_xyz

        # print(l3_xyz.shape, l3_feature.shape, ' <<< l3 xyz and feature shape')
        l2_feature = self.fp3(l2_xyz, l3_xyz, l2_points, l3_feature)
        l1_feature = self.fp2(l1_xyz, l2_xyz, l1_points, l2_feature)
        # print(l1_xyz.shape, l1_feature.shape, ' <<< l1 xyz and feature shape')
        # print(xyz.shape, l0_feature.shape, ' <<< l0 xyz and feature shape')
        x = self.fp1(xyz, l1_xyz, l0_feature, l1_feature)  # 个人认为dropout no use
        # print(x.shape, ' <<< result xyz and feature shape')
        return {'value': x}

    def _before_forward(self, input):
        # print(input['point_set'].shape, 'before forward; TODO CHECK IT')
        if self.mode == 'train':
            points = input['point_set'].cpu().data.numpy()
            # points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.from_numpy(points)
            input['point_set'] = points.type_as(input['point_set'])
            # print('before forwrad')
        return input

    def calculate_error(self, input, output):
        output = super(PointnetPlusPartSegHR, self).calculate_error(input, output)
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
    net = PointnetPlusPartSeg(config)
    net = net.cuda()
    net.set_params()
    # exit()
    from torchsummary import summary

    summary(net, batch_size=2, input_size=(4096, 3))
