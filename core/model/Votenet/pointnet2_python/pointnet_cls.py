import torch.nn as nn
import torch.nn.functional as F
import sys
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from pointnet_util import PointNetSetAbstraction
import ipdb

class get_model(nn.Module):
    def __init__(self, config):
        normal_channel = config.get('normal_channel', False)  # use color
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=False)

    def forward(self, xyz, end_points):
        xyz = xyz.permute(0, 2, 1)
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        # print(xyz.shape)
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # x = l3_points.view(B, 1024)
        l2_xyz = l2_xyz.permute(0, 2, 1)
        # print(l2_xyz.shape, l2_points.shape, '<< pointnet shape')
        end_points['seed_xyz'], end_points['seed_features'] = l2_xyz, l2_points
        return end_points
