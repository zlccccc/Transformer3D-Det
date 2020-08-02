import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model.task_basemodel.realmodel.cls_point_plane_model_rotate import cls_plane_module_rotate
from .pointnet_util import PointNetSetAbstraction
from core.model.PointnetYanx27 import provider


class PointnetPlusPointPlaneRotate(cls_plane_module_rotate):
    def __init__(self, config):
        super(PointnetPlusPointPlaneRotate, self).__init__(config)
        in_channel = 6 if config.normal_channel else 3
        self.normal_channel = config.normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256 * 3)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256 * 3, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.drop3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(512, config.num_output)

    def __forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.fc2(x)
        plane = x.view(x.shape[0], 256, 3)  # 256 points
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x, plane

    def _forward(self, input):
        # print(xyz.shape)
        xyz = input['point_set'].permute(0, 2, 1)
        x, plane = self.__forward(xyz)
        output = {'value': x, 'plane': plane}
        if 'point_set_shift' in input.keys():
            xyz_shift = input['point_set_shift'].permute(0, 2, 1)
            x_shift, plane_shift = self.__forward(xyz_shift)
            output['value_shift'] = x_shift
            output['plane_shift'] = plane_shift
        return output

    def _before_forward(self, input):
        input = super()._before_forward(input)
        if self.mode == 'train':
            points = input['point_set'].cpu().data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.from_numpy(points)
            if input['point_set'].is_cuda:
                points = points.cuda()
            input['point_set'] = points
            # print('before forwrad')
        return input


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
    net = PointnetPlusPointPlaneRotate(config)
    net = net.cuda()
    net.set_params()
    # exit()
    from torchsummary import summary

    summary(net, batch_size=2, input_size=(4096, 3))
