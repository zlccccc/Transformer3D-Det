import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model.task_basemodel.cls_model import cls_module
from .pointnet_util import PointNetSetAbstraction
from core.model.PointnetYanx27 import provider
from core.utils.utils import ensure_sub_dir


class PointnetPlusSSGInitialSaveFeature(cls_module):
    def __init__(self, config):
        super(PointnetPlusSSGInitialSaveFeature, self).__init__()
        in_channel = 6 if config.normal_channel else 3
        self.normal_channel = config.normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, config.num_output)

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
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x, l1_xyz, l1_points

    def _forward(self, input):
        # print(xyz.shape)
        xyz = input['point_set'].permute(0, 2, 1)
        value, l1_point, l1_feature = self.__forward(xyz)
        output_name = input['relativepath']
        for i in range(len(output_name)):
            dirpath = '/data1/zhaolichen/data/featuredata/' + output_name[i]
            point = l1_point[i, :, :].permute(1, 0)
            feature = l1_feature[i, :, :].permute(1, 0)
            ensure_sub_dir(dirpath)
            f = open(dirpath, 'w')
            out_mat = torch.cat([point, feature], dim=1)
            for line in range(len(out_mat)):
                print(' '.join([str(round(x, 5)) for x in out_mat[line].detach().cpu().numpy()]), file=f)
            f.close()
            cls = input['cls'][i]
            f = open('/data1/zhaolichen/data/featuredata/modelnet40_normal_resampled/label.txt', 'a')
            print(int(cls), output_name[i], file=f)
            f.close()
            # print(dirpath, point.shape, feature.shape, out_mat.shape, cls)
        input['value'] = value
        return input

    def _before_forward(self, input):
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
    net = PointnetPlusSSGInitial(config)
    net = net.cuda()
    net.set_params()
    # exit()
    from torchsummary import summary

    summary(net, batch_size=2, input_size=(4096, 3))
