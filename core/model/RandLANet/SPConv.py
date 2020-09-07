import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import pytorch_utils as pt_utils
from .utils.helper_tool import DataProcessing as DP
import numpy as np
from sklearn.metrics import confusion_matrix
import spconv
from spconv import SparseConvTensor as sptensor  # sptensor object


class ConfigSemanticKITTI:
    sub_grid_size = 0.06
    voxel_generator_config = {
        'voxel_size' : [0.4, 0.4, 0.12],
        # 'voxel_size' : [1, 1, 1],
        'point_cloud_range' : [-88, -88, -32, 88, 88, 4],
        'max_num_points' : 30,
        'max_voxels' : 50000,
    }
    output_size = 19

def Conv3d(inchannel, outchannel, kernelsize=3, stride=None, padding=None, indice_key=None, batchnorm=True, relu=nn.ReLU, conv=spconv.SparseConv3d):
    seq = []
    conv_dict = {
        'in_channels': inchannel,
        'out_channels': outchannel,
        'kernel_size': kernelsize,
        'indice_key': indice_key,
    }
    if stride is not None:
        conv_dict['stride'] = stride
    if padding is not None:
        conv_dict['padding'] = padding
    seq.append(conv(**conv_dict))
    if batchnorm:
        seq.append(nn.BatchNorm1d(outchannel))
    if relu is not None:
        seq.append(relu(outchannel))
    return spconv.SparseSequential(*seq)


class SPConv(nn.Module):
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
        self.class_weights = DP.get_class_weights(dataset_name)
        self.conv0 = nn.Sequential(
            Conv3d(3, 16, 3, conv=spconv.SubMConv3d, indice_key='l0'),
        )
        self.sa0 = nn.Sequential(
            Conv3d(16, 32, 3, conv=spconv.SubMConv3d, indice_key='l0'),
            Conv3d(32, 32, 3, conv=spconv.SubMConv3d, indice_key='l0'),
            Conv3d(32, 32, 3, 2, indice_key='l00'),
        )
        self.sa1 = nn.Sequential(
            Conv3d(32, 64, 3, conv=spconv.SubMConv3d, indice_key='l1'),
            Conv3d(64, 64, 3, conv=spconv.SubMConv3d, indice_key='l1'),
            Conv3d(64, 64, 3, 2, indice_key='l10'),
        )
        self.sa2 = nn.Sequential(
            Conv3d(64, 128, 3, conv=spconv.SubMConv3d, indice_key='l2'),
            Conv3d(128, 128, 3, conv=spconv.SubMConv3d, indice_key='l2'),
            Conv3d(128, 128, 3, 2, indice_key='l20'),
        )
        self.sa3 = nn.Sequential(
            Conv3d(128, 256, 3, conv=spconv.SubMConv3d, indice_key='l3'),
            Conv3d(256, 256, 3, 2, indice_key='l30')
        )
        self.fp3 = nn.Sequential(
            Conv3d(256, 256, 3, conv=spconv.SparseInverseConv3d, indice_key='l30'), 
            Conv3d(256, 128, 3, conv=spconv.SubMConv3d, indice_key='l3'),
        )
        self.fp2 = nn.Sequential(
            Conv3d(128 + 128, 128, 3, conv=spconv.SparseInverseConv3d, indice_key='l20'),
            Conv3d(128, 64, 3, conv=spconv.SubMConv3d, indice_key='l2'),
        )
        self.fp1 = nn.Sequential(
            Conv3d(64 + 64, 64, 3, conv=spconv.SparseInverseConv3d, indice_key='l10'),
            Conv3d(64, 64, 3, conv=spconv.SubMConv3d, indice_key='l1'),
        )
        self.fp0 = nn.Sequential(
            Conv3d(64 + 32, 64, 3, conv=spconv.SparseInverseConv3d, indice_key='l00'),
            Conv3d(64, 32, 3, conv=spconv.SubMConv3d, indice_key='l0'),
        )
        self.fc = nn.Sequential(
            Conv3d(32 + 16, 32, 3, conv=spconv.SubMConv3d, indice_key='l0'),
            Conv3d(32, self.config.output_size, 3, conv=spconv.SubMConv3d, indice_key='l0', relu=None, batchnorm=False)
        )
        self.concat = spconv.JoinTable()

    def generate_voxel(self, xyz, features):
        # self.voxel_generator = spconv.utils.VoxelGenerator(**self.config.voxel_generator_config)
        B, N, C = xyz.shape
        # print(torch.min(xyz, dim=1)[0].cpu(), '\n', torch.max(xyz, dim=1)[0].cpu(), '<< xyz min and max\n')
        # print(torch.max(xyz, dim=1)[0].cpu()-torch.min(xyz, dim=1)[0].cpu(), ' << del')
        C_range = self.config.voxel_generator_config['point_cloud_range']
        assert len(C_range) == C * 2, 'channel size not fit'
        min_position = torch.from_numpy(np.array(C_range[:C])).type_as(xyz)
        max_position = torch.from_numpy(np.array(C_range[C:])).type_as(xyz)
        voxel_size = torch.from_numpy(np.array(self.config.voxel_generator_config['voxel_size'])).type_as(xyz)
        voxel_maxposition = ((max_position - min_position) / voxel_size).int()
        # initialprint(min_position, voxel_size, ' < minpos; voxelsizepos')
        xyz = ((xyz - min_position) / voxel_size).int()  # ???
        indices = torch.arange(0, B).view(B, 1, 1).repeat([1, N, 1])
        indices = indices.type_as(xyz)
        coors = torch.cat([indices, xyz], dim=-1)
        # print(features.shape, xyz.shape)
        # print(coors, '<< coors')
        features = features.reshape(-1, features.shape[-1])
        coors = coors.reshape(-1, coors.shape[-1])
        # print(torch.min(coors, dim=0)[0].cpu(), ' ; ', torch.max(coors, dim=0)[0].cpu(), '<< fin max and min', voxel_maxposition.cpu().tolist(), flush=True)
        return coors, features, voxel_maxposition.cpu().tolist()

    @staticmethod
    def output_sptensor(feat, name):
        print(type(feat), feat.features.shape, feat.indices.shape, feat.spatial_shape, '<<', name)

    @staticmethod
    def output_spdict(feat, name):
        print(feat.indice_dict.keys(), '<< indice_dict', feat.spatial_shape)
        for key, value in sorted(feat.indice_dict.items()):
            # print(key, value[0], value[1], ' << shape')
            print(key, value[0].shape, value[1].shape, ' << shape', flush=True)

    def forward(self, inputs):
        features = inputs['features'].permute(0, 2, 1)  # Batch*channel*npoints
        xyz = inputs['xyz'][0]
        B, N, C = xyz.shape
        coors, features, voxel_maxposition = self.generate_voxel(xyz, features)
        initial = sptensor(features, coors.cpu(), voxel_maxposition, B)
        l0_down = self.conv0(initial)
        # print(features.shape, coors.shape, voxel_maxposition, '<< build tensor')
        # self.output_sptensor(l0_down, 'l0_down')
        l1_down = self.sa0(l0_down)
        l2_down = self.sa1(l1_down)
        l3_down = self.sa2(l2_down)
        l4_down = self.sa3(l3_down)
        # self.output_spdict(l4_down, 'feat')
        l3_up = self.fp3(l4_down)
        l3_feat = self.concat([l3_down, l3_up])
        l2_up = self.fp2(l3_feat)
        l2_feat = self.concat([l2_down, l2_up])
        l1_up = self.fp1(l2_feat)
        l1_feat = self.concat([l1_down, l1_up])
        l0_up = self.fp0(l1_feat)
        l0_feat = self.concat([l0_down, l0_up])
        feat = self.fc(l0_feat)
        # self.output_sptensor(l0_feat, 'l0_feat')
        # print(type(feat), feat.indices.shape, feat.indice_dict, '<< feat dict')
        # exit()
       	point_result = feat.features
        # print(point_result.shape, 'point shape', flush=True)
        point_result = point_result.reshape(B, N, -1)
        output = {}
        output['logits'] = point_result
        return output

