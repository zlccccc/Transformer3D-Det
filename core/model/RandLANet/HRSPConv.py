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


def Conv3d(inchannel, outchannel, kernelsize=3, stride=None, padding=None, indice_key=None, batchnorm=True, relu=nn.PReLU, conv=spconv.SparseConv3d):
    seq = []
    # multiply a value(as input is sparse) ? no use ?
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
    if padding is None and stride is not None and stride != 1:  # self: subm3d
        padding = (kernelsize - 1) // 2  # padding * 2 = kernel - 1
    if padding is not None:
        conv_dict['padding'] = padding
    seq.append(conv(**conv_dict))
    if batchnorm:
        seq.append(nn.BatchNorm1d(outchannel))
    if relu is not None:
        seq.append(relu(outchannel))
    return spconv.SparseSequential(*seq)


class HighResolutionBlock(nn.Module):
    def __init__(self, in_channels, out_length, way='fusion'):
        print('start building layer', in_channels, out_length)
        assert abs(len(in_channels) - out_length) <= 1
        super(HighResolutionBlock, self).__init__()
        self.fc_dict = nn.ModuleDict()
        self.moduledict = nn.ModuleDict()
        self.in_channels = in_channels  # for feature
        self.final_channels = [0 for i in range(out_length)]
        fc_channels = [32, 64, 128, 256]
        for id, channel in enumerate(in_channels):
            module_name = "%d" % id
            if id < len(fc_channels):
                _in, fin = in_channels[id], fc_channels[id]
                nowmodel = Conv3d(_in, fin, conv=spconv.SubMConv3d)
            else:
                raise NotImplementedError('HRNet: Self-Extension Not Implemented')
            self.fc_dict[module_name] = nowmodel

        for id, channel in enumerate(in_channels):  # after
            for out_id in range(out_length):
                module_name = "%d_%d" % (id, out_id)
                in_channel = fc_channels[id]  # input channel(after fc)
                nowmodel, fin = self._make_block_layer(id, out_id, in_channel, module_name, way)
                if nowmodel is not None:
                    self.moduledict[module_name] = nowmodel
                    self.final_channels[out_id] += fin

    def _make_block_layer(self, layer_in_id: int, layer_out_id: int, in_channel, module_name, way):
        fc_channels = [32, 64, 128, 256]
        _in = fc_channels[layer_in_id]
        if layer_in_id == layer_out_id:  # self mlp none?
            out_channels = [32, 64, 128, 256]
            if layer_in_id < len(out_channels):
                fin = out_channels[layer_in_id]
                nowmodel = Conv3d(_in, fin, 3, conv=spconv.SubMConv3d)
                out_channels[layer_out_id] += fin
            else:
                raise NotImplementedError('HRNet: Self-Extension Not Implemented %d' % layer_in_id)
        elif layer_in_id + 1 == layer_out_id:  # downsample
            keyword = "%d%d" % (layer_in_id, layer_out_id)
            out_channels = [32, 64, 128, 256]
            if layer_out_id < len(out_channels):
                fin = out_channels[layer_in_id]
                nowmodel, fin = Conv3d(_in, fin, 3, 2, indice_key=keyword), 64
            else:
                raise NotImplementedError('HRNet: downsample way %s layer %s Not Implemented' % (way, module_name))
        elif layer_in_id - 1 == layer_out_id:  # upsample
            keyword = "%d%d" % (layer_out_id, layer_in_id)
            out_channels = [32, 64, 128, 256]
            if layer_out_id < len(out_channels):
                fin = out_channels[layer_in_id]
                nowmodel, fin = Conv3d(_in, fin, 3, 2, conv=spconv.SparseInverseConv3d, indice_key=keyword), 64
            else:
                raise NotImplementedError('HRNet: upsample way %s layer %s Not Implemented' % (way, module_name))
        else:
            return None, 0
        print('making_block_layer', layer_in_id, layer_out_id, in_channel, way, type(nowmodel))
        return nowmodel, fin

    def _forward_layer(self, layer_in_id, layer_out_id, model, layer_in_dict, layer_out_dict):
        # print('forwarding %d->%d' % (layer_in_id, layer_out_id), type(model))
        if layer_in_id == layer_out_id:
            layer_out_dict['final_features'].append(model(layer_in_dict['features']))
        elif layer_in_id - 1 == layer_out_id or layer_in_id + 1 == layer_out_id:
            input = layer_in_dict['sparse_input']
            layer_out_dict.append(model(input))
        else:
            raise NotImplementedError('_forward layer',layer_in_id, layer_out_id)
        pass

    def forward(self, inputlist):  # forward; inputlist
        # points should in args(will generate new)
        assert len(inputlist) == len(self.in_channels)
        for id, dic in enumerate(inputlist):
            assert dic['features'].features.shape[-1] == self.in_channels[id], 'id %d input_shape not right' % id
        for key, layer in self.fc_dict.items():
            id = int(key)  # update key layer
            inputlist[id]['features'] = layer(inputlist[id]['features'])
        # for i in range(len(inputlist)):
        #     print('layer %d featuresize' % i, inputlist[i]['features'].shape)
        for _ in range(len(self.in_channels), len(self.final_channels)):
            inputlist.append({})
        for _ in range(len(self.final_channels)):
            inputlist[_]['final_features'] = []
        for key, layer in self.moduledict.items():
            in_layer, out_layer = map(int, key.split('_'))
            self._forward_layer(in_layer, out_layer, layer, inputlist[in_layer], inputlist[out_layer])
        for id in range(len(self.final_channels)):
            inputlist[id]['features'] = torch.cat(inputlist[id]['final_features'], dim=-1)
            del inputlist[id]['final_features']
        # for i, val in enumerate(self.final_channels):
        #     assert inputlist[i]['features'].shape[-1] == val, 'output_shape should be smae'
        #     print('layer %d final_feature' % i, inputlist[i]['features'].shape)
        return inputlist[: len(self.final_channels)]  # may remove some


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
                'voxel_size': [0.3, 0.3, 0.12],
                # 'voxel_size' : [1, 1, 1],
                'point_cloud_range': [-90, -90, -40, 90, 90, 20],
                'max_num_points': 30,
                'max_voxels': 50000,
            }
        else:
            raise NotImplementedError(dataset_name)
        self.class_weights = DP.get_class_weights(dataset_name)
        self.conv0 = nn.Sequential(
            Conv3d(feature_channel + 1, 8, 1, conv=spconv.SubMConv3d),
        )
        self.fc = nn.Sequential(
            Conv3d(32, 32, 1, conv=spconv.SubMConv3d),
            Conv3d(32, self.config.num_classes, 1, conv=spconv.SubMConv3d, relu=None, batchnorm=False)
        )
        self.concat = spconv.JoinTable()
        self.add = spconv.AddTable()
        self.layer0 = HighResolutionBlock([8], 2)
        self.layer1 = HighResolutionBlock(self.layer0.final_channels, 3)
        self.layer2 = HighResolutionBlock(self.layer1.final_channels, 4)
        self.layer3 = HighResolutionBlock(self.layer2.final_channels, 4)
        self.layer4 = HighResolutionBlock(self.layer3.final_channels, 3)
        self.layer5 = HighResolutionBlock(self.layer4.final_channels, 2)
        self.layer6 = HighResolutionBlock(self.layer5.final_channels, 1)

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

    @staticmethod
    def output_spfeat(feat, name):
        print(name, feat.features.shape, '<< shape', feat.spatial_shape, feat.features.detach().std().cpu(), feat.features.detach().max().cpu(), ' <<< std and max')

    def forward(self, inputs):
        features = inputs['features'].permute(0, 2, 1)  # Batch*channel*npoints
        xyz = inputs['xyz'][0]
        B, N, C = xyz.shape
        coors, features, voxel_maxposition = self.generate_voxel(xyz, features)
        initial = sptensor(features, coors.cpu(), voxel_maxposition, B)
        init_feat = self.conv0(initial)
        feat0 = self.layer0([{'feature':init_feat}])
        # print(len(feat0))
        feat1 = self.layer1(feat0)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        feat5 = self.layer5(feat4)
        feat6 = self.layer6(feat5)
        # self.output_sptensor(l0_feat, 'l0_feat')
        # print(type(feat), feat.indices.shape, feat.indice_dict, '<< feat dict')
        # exit()
        final = self.fc(feat6[0]['feature'])
        point_result = final.features
        # print(point_result.shape, 'point shape', flush=True)
        point_result = point_result.reshape(B, N, -1)
        output = {}
        output['logits'] = point_result.permute(0, 2, 1)
        return output
