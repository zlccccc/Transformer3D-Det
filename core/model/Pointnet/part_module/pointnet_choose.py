from models.part_module.sampling_utils import *
import torch.nn as nn
import torch.nn.functional as F


# F0016_DI02WH_F3D.obj is wrong; (maybe eps not okay)
class PointNetMSGChosenPReLU(nn.Module):
    def __init__(self, radius_list, nsample_list, in_channel, mlp_list, final_list, BatchNorm2d=nn.BatchNorm2d):
        super(PointNetMSGChosenPReLU, self).__init__()
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
                convs.append(nn.PReLU(out_channel))
                last_channel = out_channel
            convs = nn.Sequential(*convs)
            self.conv_blocks.append(convs)
            last_channel_all += last_channel
        self.conv_last = []
        for out_channel in final_list:
            self.conv_last.append(nn.Conv1d(last_channel_all, out_channel, 1))
            self.conv_last.append(nn.BatchNorm1d(out_channel))
            self.conv_last.append(nn.PReLU(out_channel))
            last_channel_all = out_channel
        self.conv_last = nn.Sequential(*self.conv_last)

    def forward(self, xyz, new_xyz, features=None):
        """
        Input:
            xyz: input points position data, [B, N, C]
            features: input points features, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_features: sample points feature data, [B, S, D']
        """
        B, N, C = xyz.shape
        S = new_xyz.shape[1]
        # torch.cuda.empty_cache()
        # print(new_xyz, new_xyz.shape)
        new_features_list = []
        for i, radius in enumerate(self.radius_list):
            # get k points and their features
            K = self.nsample_list[i]
            # torch.cuda.empty_cache()
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            # torch.cuda.empty_cache()
            grouped_xyz = index_points(xyz, group_idx).detach()
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
        new_features = new_features.permute(0, 2, 1)
        # print(new_features.shape, new_xyz.shape)
        return new_xyz, new_features

if __name__ == "__main__":
    import sys
