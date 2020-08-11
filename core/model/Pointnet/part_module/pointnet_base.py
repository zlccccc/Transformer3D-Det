from .sampling_utils import index_points, query_ball_point
import torch
import torch.nn as nn
import torch.nn.functional as F


# RELU: DROPOUT?
def MLP_List(in_channel, channels, FC=nn.Linear, BN=nn.BatchNorm2d, ReLU=nn.PReLU, dropout=None, lastReLU=False, debug_print=False):
    mlps = []
    last_channel = in_channel
    for id, out_channel in enumerate(channels):
        mlps.append(FC(last_channel, out_channel, 1))
        if debug_print:
            print('mlp_list insert fc(%d,%d,1)', type(FC))
        if id != len(channels) - 1 or lastReLU:
            if BN is not None:
                mlps.append(BN(out_channel))
            if ReLU is not None:
                mlps.append(ReLU(out_channel))
            if dropout is not None:
                assert len(dropout) > id, 'no dropout at id %d' % id
                mlps.append(nn.Dropout(dropout[id]))
        last_channel = out_channel
    return nn.Sequential(*mlps)


class PointNetMSGInputPoint(nn.Module):
    def __init__(self, radius_list, nsample_list, in_channel, mlp_list, final_list, BatchNorm2d=nn.BatchNorm2d, BatchNorm1d=nn.BatchNorm1d, ini_feature=0):
        super(PointNetMSGInputPoint, self).__init__()
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        last_channel_all = ini_feature
        for i in range(len(mlp_list)):  # nn.conv2d(1,1):second channel
            last_channel = in_channel + 3
            conv_now = MLP_List(last_channel, mlp_list[i],
                                FC=nn.Conv2d, BN=BatchNorm2d, ReLU=nn.PReLU, lastReLU=True)
            self.conv_blocks.append(conv_now)
            if len(mlp_list[i]) != 0:
                last_channel = mlp_list[i][-1]
            else:
                raise NotImplementedError('Warning : when PointNetMSGInputPoint; len(channel) have zero')
            last_channel_all += last_channel
        self.conv_last = MLP_List(last_channel_all, mlp_list[i],
                                  FC=nn.Conv1d, BN=BatchNorm1d, ReLU=nn.PReLU, lastReLU=True)

    def forward(self, xyz, new_xyz, features=None, new_features=None):
        """
        Input:
            xyz: input points position data, [B, N, C]
            features: input points features, [B, N, D]
            new_xyz: sampled points position data, [B, S, C]
            new_features: sample points feature data, [B, S, D']
        Return:
            new_features: final features, [B, S, D'']
        """
        B, N, C = xyz.shape
        S = new_xyz.shape[1]
        # torch.cuda.empty_cache()
        # print(new_xyz, new_xyz.shape)
        if new_features is not None:
            new_features_list = [new_features.permute(0, 2, 1)]
        else:
            new_features_list = []
        for i, radius in enumerate(self.radius_list):
            # get k points and their features
            # torch.cuda.empty_cache()
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            # print(group_idx)
            # print(np.max(group_idx), np.min(group_idx), xyz.shape, group_idx)
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

        # for _ in new_features_list:
        #     print(_.shape)
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
        return new_features
