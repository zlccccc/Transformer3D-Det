from .sampling_utils import farthest_point_sample, random_point_sample, index_points
from .pointnet_base import PointNetMSGInputPoint
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetMSG(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, final_list,
                 BatchNorm2d=nn.BatchNorm2d, ReLU=nn.PReLU):
        super(PointNetMSG, self).__init__()
        self.npoint = npoint
        self.pointnet = PointNetMSGInputPoint(radius_list, nsample_list, in_channel, mlp_list, final_list,
                                              BatchNorm2d=BatchNorm2d, ReLU=ReLU)

    def forward(self, xyz, features=None):
        """
        Input:
            xyz: input points position data, [B, N, C]
            features: input points features, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_features: sample points feature data, [B, S, D']
        """
        B, N, C = xyz.shape
        S = self.npoint
        if self.npoint != N:
            fpx_idx = farthest_point_sample(xyz, S)
            new_xyz = index_points(xyz, fpx_idx)
        else:
            new_xyz = xyz
        new_features = self.pointnet(xyz, new_xyz, features, None)
        return new_xyz, new_features


class PointNetMSGRandomSample(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, final_list,
                 BatchNorm2d=nn.BatchNorm2d, ReLU=nn.PReLU):
        super(PointNetMSGRandomSample, self).__init__()
        self.npoint = npoint
        self.pointnet = PointNetMSGInputPoint(radius_list, nsample_list, in_channel, mlp_list, final_list, BatchNorm2d)

    def forward(self, xyz, features=None, static=None):
        """
        Input:
            xyz: input points position data, [B, N, C]
            features: input points features, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_features: sample points feature data, [B, S, D']
        """
        B, N, C = xyz.shape
        S = self.npoint
        if self.npoint != N:
            fpx_idx = random_point_sample(xyz, S)
            new_xyz = index_points(xyz, fpx_idx)
        else:
            new_xyz = xyz
        new_features = self.pointnet(xyz, new_xyz, features, None)
        return new_xyz, new_features


if __name__ == "__main__":
    import sys
