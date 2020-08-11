from .pointnet_downsample import PointNetMSG, PointNetMSGRandomSample
from .pointnet_upsample import PointNetPropagation
from .pointnet_base import MLP_List

import torch
import torch.nn as nn


class PointNetFeature(nn.Module):
    def __init__(self, in_channel, mlp, fc, dropout=None, BatchNorm2d=nn.BatchNorm2d, BatchNorm1d=nn.BatchNorm1d):
        super(PointNetFeature, self).__init__()
        self.convs = []
        last_channel = in_channel + 3
        self.convs = MLP_List(last_channel, mlp,
                              FC=nn.Conv2d, BN=BatchNorm2d, ReLU=nn.PReLU, lastReLU=True)
        if len(mlp) != 0:
            last_channel = mlp[-1]
        if len(fc) == 0:
            raise NotImplementedError('len(fc)!=0')
        if dropout is not None and len(dropout) == 0:
            dropout = None
        self.fc = MLP_List(last_channel, fc,
                           FC=nn.Linear, BN=BatchNorm1d, ReLU=nn.PReLU, dropout=dropout, lastReLU=False)

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
        # print('shape final', features.shape)
        return features


if __name__ == "__main__":
    import sys

    sys.path.append('../')
    net = PointNetMSG(512, [0.1, 0.2, 0.4], [16, 32, 128], 0,
                      [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                      [256, 256])
    net = net.cuda()
    from torchsummary import summary

    summary(net, batch_size=2, input_size=(1234, 3))
