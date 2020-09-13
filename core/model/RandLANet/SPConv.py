import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import pytorch_utils as pt_utils
from .utils.helper_tool import ConfigSemanticKITTI, ConfigS3DIS, ConfigSemantic3D
from .utils.helper_tool import DataProcessing as DP
import numpy as np
from sklearn.metrics import confusion_matrix
import spconv
from spconv import SparseConvTensor as sptensor  # sptensor object


class SparseMultiply(nn.Module): # else grad may disappear
# class SparseMultiply(spconv.SparseModule): # else grad may disappear
    def __init__(self, weight, bias=0):
        super(SparseMultiply, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, feat):
        # print(type(feat))
        if isinstance(feat, torch.Tensor):
            print(feat.shape, '? sparse multiply ? std=', feat.std().detach().cpu(), flush=True)
            feat = feat * self.weight + self.bias
        else:
            feat.features = feat.features * self.weight + self.bias
        return feat


def Conv3d(inchannel, outchannel, kernelsize=3, stride=None, padding=None, indice_key=None, batchnorm=True, relu=nn.PReLU, conv=spconv.SparseConv3d, multiply=1, bias=0):
    seq = []
    # multiply a value(as input is sparse) ? no use ?
    if multiply is None: # as a surface; not useful
        multiply = kernelsize
    if multiply != 1 or bias != 0:
        seq.append(SparseMultiply(multiply, bias))
    conv_dict = {
        'in_channels': inchannel,
        'out_channels': outchannel,
        'kernel_size': kernelsize,
        'indice_key': indice_key,
    }
    if stride is not None:
        conv_dict['stride'] = stride
    # VALID; fin = (initial + padding * 2 - kernel + stride) / stride; to make last available
    # SAME;  fin = (initial + stride - 1) / stride
    if padding is None and stride is not None and stride != 1: # self: subm3d
        padding = (kernelsize - 1) // 2  # padding * 2 = kernel - 1
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
            self.config.voxel_generator_config = {
                            'voxel_size' : [0.3, 0.3, 0.12],
                            # 'voxel_size' : [1, 1, 1],
                            'point_cloud_range' : [-90, -90, -40, 90, 90, 20],
                            'max_num_points' : 30,
                            'max_voxels' : 50000,
                        }
        else:
            raise NotImplementedError(dataset_name)
        self.class_weights = DP.get_class_weights(dataset_name)
        self.conv0 = nn.Sequential(
            Conv3d(4, 32, 1, conv=spconv.SubMConv3d, indice_key='l0'),  # l0 not exist
            Conv3d(32, 32, 1, conv=spconv.SubMConv3d, indice_key='l0'),
        )
        self.sa0 = nn.Sequential(
            Conv3d(32, 32, 1, conv=spconv.SubMConv3d, indice_key='l0'),
            Conv3d(32, 64, 2, 2, indice_key='l00'),
        )
        self.sa1 = nn.Sequential(
            Conv3d(64, 64, 3, conv=spconv.SubMConv3d, indice_key='l1'),
            Conv3d(64, 128, 3, conv=spconv.SubMConv3d, indice_key='l1'),
            Conv3d(128, 128, 3, 2, indice_key='l10'),
        )
        self.sa2 = nn.Sequential(
            Conv3d(128, 128, 3, conv=spconv.SubMConv3d, indice_key='l2'),
            Conv3d(128, 256, 3, conv=spconv.SubMConv3d, indice_key='l2'),
            Conv3d(256, 256, 3, 2, indice_key='l20'),
        )
        self.sa3 = nn.Sequential(
            Conv3d(256, 512, 3, conv=spconv.SubMConv3d, indice_key='l3'),
            Conv3d(512, 512, 3, 2, indice_key='l30'),
            Conv3d(512, 512, 3, conv=spconv.SubMConv3d, indice_key='l4'),
            Conv3d(512, 512, 3, conv=spconv.SubMConv3d, indice_key='l4'),
        )
        self.fp3 = nn.Sequential(
            Conv3d(512, 512, 3, conv=spconv.SparseInverseConv3d, indice_key='l30'), 
            Conv3d(512, 256, 3, conv=spconv.SubMConv3d, indice_key='l3'),
        )
        self.fp2 = nn.Sequential(
            Conv3d(256 + 256, 256, 3, conv=spconv.SparseInverseConv3d, indice_key='l20'),
            Conv3d(256, 128, 3, conv=spconv.SubMConv3d, indice_key='l2'),
        )
        self.fp1 = nn.Sequential(
            Conv3d(128 + 128, 128, 3, conv=spconv.SparseInverseConv3d, indice_key='l10'),
            Conv3d(128, 64, 3, conv=spconv.SubMConv3d, indice_key='l1'),
        )
        self.fp0 = nn.Sequential(
            Conv3d(64 + 64, 64, 2, conv=spconv.SparseInverseConv3d, indice_key='l00'),
            Conv3d(64, 32, 1, conv=spconv.SubMConv3d, indice_key='l0'),
        )
        self.fc = nn.Sequential(
            Conv3d(32 + 32, 32, 1, conv=spconv.SubMConv3d, indice_key='l0'),
            Conv3d(32, self.config.num_classes, 1, conv=spconv.SubMConv3d, indice_key='l0', relu=None, batchnorm=False)
        )
        self.concat = spconv.JoinTable()
        self.add = spconv.AddTable()

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
        features = features / (max_position - min_position) * 2  # normalize
        feature_cat = torch.ones([features.shape[0], 1]).type_as(features)
        features = torch.cat([feature_cat, features], dim=-1)
        # print(features.shape, feature_cat.shape, '<< input feature shape')
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
            # inid(block); outid(block); indice_pairs, indice_pair_num, out_shape
            print(key, len(value), value[0].shape, value[1].shape, ' << shape;', flush=True, end='')
            # print(np.array(value[2]).shape, value[3], value[4], flush=True)
            # print(value[2][:, :, :10], ' << val 2 indice compare')
            print(np.array(value[2]).shape, value[4], flush=True)
            # pair: x-y link; next lines not useful
            ### pair_shape = np.array(value[2])!=-1
            ### pair_shape = np.sum(pair_shape, axis=1)
            ### print(value[0][-4:], value[1][-4:], ' << compare')
            ### nonzeroshape = np.sum(pair_shape != 0, axis=-1)
            ### print('  zeros 0', value[0][nonzeroshape[0]-4:nonzeroshape[0]+4])
            ### print('  zeros 1', value[1][nonzeroshape[0]-4:nonzeroshape[0]+4])
            ### print('  zeros', value[2][0][:, nonzeroshape[0]-4:nonzeroshape[0]+4])
            ### print(' UPSAMPLE MAXIMIZE', np.max(np.array(value[2])), np.sum(pair_shape))
            ### print('   upsample shape,', np.sum(pair_shape != 0, axis=-1), np.mean(pair_shape), ' <<< downsample', pair_shape.shape, value[3])

    @staticmethod
    def output_spfeat(feat, name):
        print(name, feat.features.shape, '<< shape', feat.spatial_shape, feat.features.detach().std().cpu(), feat.features.detach().max().cpu(), ' <<< std and max')

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
        # self.output_spfeat(initial, 'initial')
        # self.output_spfeat(l0_down, 'l0down')
        # self.output_spfeat(l1_down, 'l1down')
        # self.output_spfeat(l2_down, 'l2down')
        # self.output_spfeat(l3_down, 'l3down')
        # self.output_spfeat(l4_down, 'l4down')
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
        output['logits'] = point_result.permute(0, 2, 1)
        return output

