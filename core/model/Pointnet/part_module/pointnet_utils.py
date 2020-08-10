from .pointnet_downsample import PointNetMSG, PointNetMSGRandomSample
from .pointnet_upsample import PointNetPropagation

import torch
import torch.nn as nn


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
            self.convs.append(nn.PReLU(out_channel))
            last_channel = out_channel
        self.convs = nn.Sequential(*self.convs)
        self.fc = []
        for id, out_channel in enumerate(fc):
            self.fc.append(nn.Linear(last_channel, out_channel))
            if id != len(fc) - 1:
                self.fc.append(BatchNorm1d(out_channel))  # its shape is ok
                self.fc.append(nn.PReLU(out_channel))
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
