import torch
import torch.nn as nn
from core.model.task_basemodel.taskmodel.seg_model import seg_module
from core.model.Pointnet.part_module.pointnet_utils import PointNetMSG, PointNetFeature, PointNetPropagation, MLP
from core.model.PointnetYanx27 import provider
from core.model.task_error.ShapeNetError import ShapeNetError


class HighResolutionBlock(nn.Module):
    def __init__(self, in_channels, out_length, way='fusion'):
        print('start building layer', in_channels, out_length)
        assert abs(len(in_channels) - out_length) <= 1
        super(HighResolutionBlock, self).__init__()
        self.fc_dict = nn.ModuleDict()
        self.moduledict = nn.ModuleDict()
        self.in_channels = in_channels  # for feature
        self.final_channels = [0 for i in range(out_length)]
        fc_channels = [64, 128, 256, 1024]
        for id, channel in enumerate(in_channels):
            module_name = "%d" % id
            if id < len(fc_channels):
                _in, fin = in_channels[id], fc_channels[id]
                nowmodel = MLP(_in, [fin, fin], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)  # self64+down128
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
        if layer_in_id == layer_out_id:  # self mlp none?
            out_channels = [64, 128, 256]
            if layer_in_id < len(out_channels):
                fin = out_channels[layer_in_id]
                nowmodel = MLP(in_channel, [fin, fin], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)  # self64+down128
                out_channels[layer_out_id] += fin
            else:
                raise NotImplementedError('HRNet: Self-Extension Not Implemented %d' % layer_in_id)
        elif layer_in_id + 1 == layer_out_id:
            if layer_in_id == 0:
                nowmodel, fin = PointNetMSG(512, [0.2], [32], in_channel, [[64, 128]], [128]), 128
            elif layer_in_id == 1:
                nowmodel, fin = PointNetMSG(128, [0.4], [64], in_channel, [[128, 256]], [256]), 256
            elif layer_in_id == 2:
                nowmodel, fin = PointNetFeature(in_channel, [256, 512, 1024], [1024]), 1024
            else:
                raise NotImplementedError('HRNet: downsample way %s layer %s Not Implemented' % (way, module_name))
        elif layer_in_id - 1 == layer_out_id:
            if layer_in_id == 1 or layer_in_id == 2 or layer_in_id == 3:
                nowmodel, fin = PointNetPropagation(in_channel=in_channel, mlp=[in_channel]), in_channel
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
        elif layer_in_id + 1 == layer_out_id:  # downsample points
            xyz, features = layer_in_dict['points'], layer_in_dict['features']
            # PointNetMSG
            if isinstance(model, PointNetMSG):
                # def forward(self, xyz, features=None, return_group_id=False, grouped_ids=None, sampled_new_xyz=None):
                if 'points' not in layer_out_dict.keys():
                    new_xyz, (new_feature, group_id) = model(xyz, features, return_group_id=True)
                    layer_out_dict['points'], layer_out_dict['group_id'] = new_xyz, group_id
                else:
                    sampled_new_xyz, group_id = layer_out_dict['points'], layer_out_dict['group_id']
                    new_xyz, new_feature = model(xyz, features, grouped_ids=group_id, sampled_new_xyz=sampled_new_xyz)
                layer_out_dict['final_features'].append(new_feature)
            elif isinstance(model, PointNetFeature):
                if 'points' not in layer_out_dict.keys():
                    layer_out_dict['points'] = torch.zeros((xyz.shape[0], 1, 3)).type_as(xyz)
                    new_feature = model(xyz, features)
                    new_feature = new_feature.view(new_feature.shape[0], 1, new_feature.shape[1])
                layer_out_dict['final_features'].append(new_feature)
                # print('PointNetFeature: TODO')
            else:
                raise NotImplementedError(type(model))
        elif layer_in_id - 1 == layer_out_id:
            if isinstance(model, PointNetPropagation):
                xyz, features = layer_in_dict['points'], layer_in_dict['features']
                new_xyz = layer_out_dict['points']  # upsample and not self
                new_feature = model(new_xyz, xyz, None, features)
                layer_out_dict['final_features'].append(new_feature)
            else:
                raise NotImplementedError(type(model))
        else:
            raise NotImplementedError('_forward layer',layer_in_id, layer_out_id)
        pass

    def forward(self, inputlist):  # forward; hstack
        # points should in args(will generate new)
        assert len(inputlist) == len(self.in_channels)
        for id, dic in enumerate(inputlist):
            assert dic['features'].shape[-1] == self.in_channels[id], 'id %d input_shape not right' % id
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
        return inputlist[: len(self.final_channels)]


class PointnetPlusPartSegHR(seg_module):
    def __init__(self, config):
        self.params = []
        self.task_type = config.get('task_type', 'No Impl')
        if self.task_type == 'ShapeNet':
            self.num_output = 50
            self.num_label = 16  # before fc (later use)
        else:
            raise NotImplementedError('task type %s' % self.task_type)

        normal_channel = config.get('normal_channel', True)
        self.config = config
        super(PointnetPlusPartSegHR, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        in_channel = in_channel + 3 + self.num_label  # initial xyz and mark
        # layer 0: 0->0,1; fc->fp and sa;
        self.layer0 = HighResolutionBlock([in_channel], 2)
        # layer 1: 0->0,1; 1->0,1,2  # nearby trans first
        self.layer1 = HighResolutionBlock(self.layer0.final_channels, 3)
        # layer 2: 0->0,1; 1->0,1,2; 2->1,2,3(3=feature)
        self.layer2 = HighResolutionBlock(self.layer1.final_channels, 4)  # TODO: CHANGE 3 TO 4
        # layer 3: 0->0,1; 1->0,1,2; 2->1,2; 3->2
        self.layer3 = HighResolutionBlock(self.layer2.final_channels, 3)
        # layer 4: 0->0,1; 1->0,1; 2->1
        self.layer4 = HighResolutionBlock(self.layer3.final_channels, 2)
        # layer 5: 0->0; 1->0
        self.layer5 = HighResolutionBlock(self.layer4.final_channels, 1)
        # layer 6: 0->0
        self.layer6 = HighResolutionBlock(self.layer5.final_channels, 1)
        self.fc = MLP(64, [self.num_output], dropout=None, FC=nn.Conv1d, BN=nn.BatchNorm1d, ReLU=nn.PReLU)
        self.init_relu = 'relu'
        self.init_params(nn.BatchNorm2d, init_type='kaiming_normal')

    def _forward(self, input):
        # print(xyz.shape)
        xyz = input['point_set']
        cls_label = input['cls']  # pointnet cls use

        B, N, _ = xyz.shape
        norm = xyz  # channel = 3 + (rgb)
        xyz = xyz[:,:,:3]

        cls_label = nn.functional.one_hot(cls_label, self.num_label).type_as(xyz)
        cls_label_one_hot = cls_label.view(B, 1, self.num_label).repeat(1, N, 1)
        xyz_feature = torch.cat([cls_label_one_hot, norm], dim=2)
        layer0_feature = [{'points': xyz, 'features': xyz_feature}]  # length
        layer1_feature = self.layer0(layer0_feature)
        layer2_feature = self.layer1(layer1_feature)
        layer3_feature = self.layer2(layer2_feature)
        layer4_feature = self.layer3(layer3_feature)
        layer5_feature = self.layer4(layer4_feature)
        layer6_feature = self.layer5(layer5_feature)
        layer7_feature = self.layer6(layer6_feature)
        x = layer7_feature[0]['features']
        x = self.fc(x)
        # print(x.shape, ' <<< result xyz and feature shape')
        return {'value': x}

    def _before_forward(self, input):
        # print(input['point_set'].shape, 'before forward; TODO CHECK IT')
        if self.mode == 'train':
            points = input['point_set'].cpu().data.numpy()
            # points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.from_numpy(points)
            input['point_set'] = points.type_as(input['point_set'])
            # print('before forwrad')
        return input

    def calculate_error(self, input, output):
        output = super(PointnetPlusPartSegHR, self).calculate_error(input, output)
        if self.task_type == 'ShapeNet':
            output = ShapeNetError(input, output)
        return output

if __name__ == "__main__":
    import sys
    import os
    from easydict import EasyDict

    config = {
        'num_output': 83,
        'normal_channel': False
    }
    config = EasyDict(config)
    print(os.getcwd())
    net = PointnetPlusPartSegHR(config)
    net = net.cuda()
    net.set_params()
    # exit()
    from torchsummary import summary

    summary(net, batch_size=2, input_size=(4096, 3))
