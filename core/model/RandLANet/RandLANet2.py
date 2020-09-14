import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import pytorch_utils as pt_utils
from .utils.helper_tool import ConfigSemanticKITTI, ConfigS3DIS, ConfigSemantic3D
from .utils.helper_tool import DataProcessing as DP
import numpy as np
from sklearn.metrics import confusion_matrix


class RandLANet(nn.Module):
    def __init__(self, config):
        super().__init__()
        dataset_name = config.get('dataset', 'SemanticKITTI')  # ONLY GET NAME
        feature_channel = config.get('feature_channel', 3)
        if dataset_name == 'Semantic3D':
            self.config = ConfigSemantic3D
        elif dataset_name == 'SemanticKITTI':
            self.config = ConfigSemanticKITTI
        else:
            raise NotImplementedError(dataset_name)
        self.dataset_name = dataset_name
        self.class_weights = DP.get_class_weights(dataset_name)

        self.fc0 = pt_utils.Conv1d(feature_channel, 8, kernel_size=1, bn=True, activation=nn.PReLU(8))

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True, activation=nn.PReLU(d_out))

        self.decoder_blocks = nn.ModuleList()
        for j in range(self.config.num_layers):
            if j != self.config.num_layers - 1:
                d_in = d_out + 2 * self.config.d_out[-j - 2]
                d_out = 2 * self.config.d_out[-j - 2]
            else:
                d_in = 4 * self.config.d_out[0]
                d_out = 2 * self.config.d_out[0]
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True, activation=nn.PReLU(d_out)))
            # print('decoder block', d_in, d_out)

        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1, 1), bn=True, activation=nn.PReLU(64))
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True, activation=nn.PReLU(32))
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1, 1), bn=False, activation=None)

    def forward(self, inputs):

        features = inputs['features']  # Batch*channel*npoints
        # print(features.std(), 'features std', flush=True)
        if self.dataset_name == 'SemanticKITTI':
            features = features / 6 # normalize
        elif self.dataset_name == 'SemanticKITTI':
            pass
        else:
            raise NotImplementedError(self.dataset_name)
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, inputs['xyz'][i], inputs['neigh_idx'][i])

            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
            features = f_sampled_i
            # print('downsample encoder feature shape', features.shape)
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
            # torch.cuda.empty_cache()
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, inputs['interp_idx'][-j - 1])
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

            features = f_decoder_i
            # print('upsample decoder feature shape', features.shape)
            f_decoder_list.append(f_decoder_i)
            # torch.cuda.empty_cache()
        # ###########################Decoder############################

        features = self.fc1(features)
        features = self.fc2(features)
        features = self.dropout(features)
        features = self.fc3(features)
        f_out = features.squeeze(3)

        output = {}
        # output['value'] = f_out.permute(0, 2, 1).clone()
        output['logits'] = f_out
        # print(f_out.shape)
        # f_out.mean().backward()
        # exit(0)
        return output

    @staticmethod
    def random_sample(feature, pool_idx): # RIGHT; GROUP DIRECTLY
        """
        :param feature: [B, d, N, 1] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, d, N', 1] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        # print(feature.shape, pool_idx.shape, pool_features.shape)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):  # nearest! it is not so good
        """
        :param feature: [B, d, N, 1] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, d, up_num_points, 1] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        # print(feature.shape, interp_idx.shape, interpolated_features.shape, flush=True)
        return interpolated_features


def compute_acc(end_points):
    logits = end_points['valid_logits']
    labels = end_points['valid_labels']
    logits = logits.max(dim=1)[1]
    if labels.shape[0] != 0:
        acc = (logits == labels).sum().float() / float(labels.shape[0])
    else:
        acc = 0
    end_points['acc'] = acc
    return acc, end_points


class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(d_in, d_out // 2, kernel_size=(1, 1), bn=True, activation=nn.PReLU(d_out // 2))
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc + shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  # d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out // 2, kernel_size=(1, 1), bn=True, activation=nn.PReLU(d_out // 2))
        self.att_pooling_1 = Att_pooling(d_out, d_out // 2)

        self.mlp2 = pt_utils.Conv2d(d_out // 2, d_out // 2, kernel_size=(1, 1), bn=True, activation=nn.PReLU(d_out // 2))
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True, activation=nn.PReLU(d_out))

    def forward(self, feature_set):
        # print(feature_set.shape, flush=True)
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg

