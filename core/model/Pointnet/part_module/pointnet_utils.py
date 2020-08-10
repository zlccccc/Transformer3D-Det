from .sampling_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp_list(in_channel, channels, FC=nn.Linear, BN=nn.BatchNorm2d, ReLU=nn.ReLU, lastReLU=False):
    mlps = []
    last_channel = in_channel
    for id, out_channel in enumerate(channels):
        mlps.append(FC(last_channel, out_channel, 1))
        if id != len(channels) - 1 or lastReLU:
            mlps.append(BN(out_channel))
            mlps.append(ReLU(out_channel))
        last_channel = out_channel
    return nn.Sequential(*mlps)


class PointNetFeature(nn.Module):
    def __init__(self, in_channel, mlp, fc, dropout, BatchNorm2d=nn.BatchNorm2d, BatchNorm1d=nn.BatchNorm1d):
        super(PointNetFeature, self).__init__()
        self.convs = []
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.convs.append(BatchNorm2d(out_channel))
            self.convs.append(nn.ReLU(out_channel))
            last_channel = out_channel
        self.convs = nn.Sequential(*self.convs)
        self.fc = []
        for id, out_channel in enumerate(fc):
            self.fc.append(nn.Linear(last_channel, out_channel))
            if id != len(fc) - 1:
                self.fc.append(BatchNorm1d(out_channel))  # its shape is ok
                self.fc.append(nn.ReLU(out_channel))
                self.fc.append(nn.Dropout(dropout[id]))
            last_channel = out_channel
        self.fc = nn.Sequential(*self.fc)

    def forward(self, xyz, features):
        """
        Input:
            xyz: input points position data, [B, N, C]
            features: input features data, [B, N, D]
        Return:
            new_features: feature, [B, S]
        """
        B, N, C = xyz.shape
        if features is None:
            features = xyz
        else:
            features = torch.cat([xyz, features], dim=-1)
        # print('shape', features.shape)  # [B, N, C+D]
        features = features.permute(0, 2, 1)  # [B, C+D, N, 1]
        features = features.view(B, -1, N, 1)
        # print(features.shape)
        features = self.convs(features)
        features = torch.max(features, 2)[0]  # [B, D', S]
        # print('shape after', features.shape)
        features = features.view(B, -1)
        features = self.fc(features)
        # print(features)
        # print('shape final', features.shape)
        return features


# F0016_DI02WH_F3D.obj is wrong; (maybe eps not okay)
class PointNetMSG(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, final_list, BatchNorm2d=nn.BatchNorm2d):
        super(PointNetMSG, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        last_channel_all = 0
        for i in range(len(mlp_list)):
            convs = []  # nn.conv2d(1,1):second channel
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                convs.append(BatchNorm2d(out_channel))
                convs.append(nn.ReLU(out_channel))
                last_channel = out_channel
            convs = nn.Sequential(*convs)
            self.conv_blocks.append(convs)
            last_channel_all += last_channel
        self.conv_last = []
        for out_channel in final_list:
            self.conv_last.append(nn.Conv1d(last_channel_all, out_channel, 1))
            self.conv_last.append(nn.BatchNorm1d(out_channel))
            self.conv_last.append(nn.ReLU(out_channel))
            last_channel_all = out_channel
        self.conv_last = nn.Sequential(*self.conv_last)

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
        # torch.cuda.empty_cache()
        fpx_idx = farthest_point_sample(xyz, S)
        # torch.cuda.empty_cache()
        new_xyz = index_points(xyz, fpx_idx)
        # print(new_xyz, new_xyz.shape)
        new_features_list = []
        for i, radius in enumerate(self.radius_list):
            # get k points and their features
            K = self.nsample_list[i]
            # torch.cuda.empty_cache()
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            # torch.cuda.empty_cache()
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if features is not None:
                grouped_points = index_points(features, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            # print(K, grouped_xyz.shape, grouped_points.shape)
            # print(grouped_points.shape)
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]; conv second channel
            # print('grouped_points', grouped_points.shape)
            grouped_points = self.conv_blocks[i](grouped_points)
            # print(grouped_points.shape)
            new_features = torch.max(grouped_points, 2)[0]  # [B, D', S]
            # print('new_features:', new_features.shape)
            new_features_list.append(new_features)  # like pointnet

        new_features = torch.cat(new_features_list, dim=1)  # for fewer reshape and permute
        # print(new_features.shape)
        B, D, N = new_features.shape
        # new_features = new_features.view(B, D, N)
        # new_features = new_features.permute(0,)
        new_features = self.conv_last(new_features)
        # new_features = new_features.view(B, -1, N)
        # print(new_features.shape)
        new_features = new_features.permute(0, 2, 1)
        # print(new_features.shape, new_xyz.shape)
        return new_xyz, new_features


class PointNetPropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetPropagation, self).__init__()
        last_channel = in_channel
        self.mlp = []
        for out_channel in mlp:
            self.mlp.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.mlp = nn.Sequential(*self.mlp)

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


if __name__ == "__main__":
    import sys

    sys.path.append('../')
    net = PointNetMSG(512, [0.1, 0.2, 0.4], [16, 32, 128], 0,
                      [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                      [256, 256])
    net = net.cuda()
    from torchsummary import summary

    summary(net, batch_size=2, input_size=(1234, 3))
