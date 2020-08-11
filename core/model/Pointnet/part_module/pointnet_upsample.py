from .sampling_utils import index_points, square_distance
from .pointnet_base import MLP_List
import torch
import torch.nn as nn


# should add one mlp_list at last
class PointNetPropagation(nn.Module):
    def __init__(self, in_channel, mlp, BatchNorm1d=nn.BatchNorm1d, ReLU=nn.PReLU):
        super(PointNetPropagation, self).__init__()
        last_channel = in_channel
        self.mlp = MLP_List(last_channel, mlp,  # lastReLU=True? checkit
                            FC=nn.Conv1d, BN=BatchNorm1d, ReLU=ReLU)  # no relu and dropout?

    def forward(self, xyz1, xyz2, features1, features2):
        """
        Input:
            xyz1: input points position data, [B, N, C]
            xyz2: sampled input points position data, [B, S, C]
            features1: input features data, [B, N, D]
            features2: input features data, [B, S, D]
        Return:
            new_features: upsampled features data, [B, D', N]
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = features2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(features2, idx) * weight.view(B, N, 3, 1), dim=2)
        # print('cat', features1.shape, interpolated_points.shape)

        if features1 is not None:
            new_features = torch.cat([features1, interpolated_points], dim=-1)
        else:
            new_features = interpolated_points

        new_features = new_features.permute(0, 2, 1)
        new_features = self.mlp(new_features)
        new_features = new_features.permute(0, 2, 1)
        return new_features
