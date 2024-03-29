import torch.nn as nn
from core.model.task_basemodel.backbone.base_model import base_module
from core.model.Pointnet import model_entry
from .utils import split_obj, split_rel
from .loss_utils import calculate_loss
from .error_utils import calculate_kth_error, calculate_recall_per_edge


class PointNetSG(base_module):
    def __init__(self, config):
        super(PointNetSG, self).__init__()
        self.params = []
        self.obj_module = None
        if 'objmodel' in config.keys():
            print('training objmodel')
            self.obj_module = model_entry(config.objmodel)
            self.obj_loss_type = config.objmodel.get('loss_type', 'focal_loss')
        self.rel_module = None
        if 'relmodel' in config.keys():
            print('training relmodel')
            self.rel_module = model_entry(config.relmodel)
            self.rel_loss_type = config.relmodel.get('loss_type', 'focal_loss')
        assert self.obj_module is not None, 'obj module is None'
        assert self.rel_module is not None, 'rel module is None'
        # print(self.obj_module.parameters())
        self.init_relu = 'relu'
        self.init_params(nn.BatchNorm2d, init_type='kaiming_normal')

    def _forward(self, input):
        # print(xyz.shape)
        output = {}
        if self.obj_module is not None:
            assert 'object_points' in input.keys(), 'dataloader should have "object points" object(or in utils.py)'
            # print('calculating object')
            output['obj_result'] = self.obj_module._forward({'point_set': input['object_points']})['value']
        if self.rel_module is not None:
            assert 'rel_points' in input.keys(), 'dataloader should have "relationship" object(or in utils.py)'
            # print('calculating relationship')
            output['rel_result'] = self.rel_module._forward({'point_set': input['rel_points']})['value']
        return output

    def _before_forward(self, input):
        if 'object_point_set' in input.keys():
            input = split_obj(input)
        if 'rel_point_set' in input.keys():
            input = split_rel(input)
        # print('relationship final keys', input.keys())
        return input

    def calculate_loss(self, input, output):
        loss = 0
        if self.rel_module is not None:
            result, target = output['rel_result'], input['one_hot_rel_target']  # one_hot_target
            rel_loss = calculate_loss(result, target, self.rel_loss_type) * 100  # weight
            output['rel_n_count(loss)'] = result.shape[0]
            output['rel_loss'] = rel_loss
            output['n_count'] = output['rel_n_count(loss)']
            output['all_rel_top_1_recall(loss)'] = calculate_kth_error(result, target, 1, 'recall')
            loss += rel_loss
        if self.obj_module is not None:
            result, target = output['obj_result'], input['object_target']
            obj_loss = calculate_loss(result, target, self.obj_loss_type) * 100  # weight
            output['obj_n_count(loss)'] = result.shape[0]
            output['obj_loss'] = obj_loss
            output['n_count'] = output['obj_n_count(loss)']
            output['obj_top_1_acc(loss)'] = calculate_kth_error(result, target, 1, 'accurancy')
            loss += obj_loss
        output['loss'] = loss
        return output

    def calculate_error(self, input, output):
        error = 1
        if self.rel_module is not None:
            result, target = output['rel_result'], input['one_hot_rel_target']
            output['all_rel_top_1_recall(error)'] = calculate_kth_error(result, target, 5, 'recall')
            rel_mask = input['rel_mask']
            if rel_mask.sum() != 0:
                mask = (rel_mask == 1)[:, 0]
                # print('mask', mask.shape)
                result = result[mask, 1:]
                target = target[mask, 1:]
                # print('testing relation', result.shape, target.shape)
                output['rel_top_1_recall(error)'] = calculate_kth_error(result, target, 1, 'recall')
                output['rel_top_3_recall(error)'] = calculate_kth_error(result, target, 3, 'recall')
                output['rel_top_5_recall(error)'] = calculate_kth_error(result, target, 5, 'recall')
                output['recall_number(error)'] = 1
            else:
                output['recall_number(error)'] = 0
                # print('DONE testing relation', result.shape, target.shape)
            output['rel_n_count(error)'] = result.shape[0]
            error *= output['all_rel_top_1_recall(error)']
        if self.obj_module is not None:
            result, target = output['obj_result'], input['object_target']
            output['obj_acc(error)'] = calculate_kth_error(result, target, 1, 'top_accurancy')
            output['obj_top_1_acc(error)'] = calculate_kth_error(result, target, 1, 'accurancy')
            output['obj_top_5_acc(error)'] = calculate_kth_error(result, target, 5, 'accurancy')
            output['obj_top_10_acc(error)'] = calculate_kth_error(result, target, 10, 'accurancy')
            output['obj_n_count(error)'] = result.shape[0]
            object_predict, object_target = result, target
            error *= output['obj_top_1_acc(error)']
        if self.rel_module is not None and self.obj_module is not None:
            # calculate all error
            # (object_predict, object_target), (rel_predict, one_hot_rel_target), object_idx, rel_idx, rel_mask, maxk
            rel_predict, one_hot_rel_target = output['rel_result'], input['one_hot_rel_target']
            output['edge_acc_50(error)'] = calculate_recall_per_edge(object_predict, object_target, rel_predict,
                                                                     one_hot_rel_target, input, 50)
            output['edge_acc_100(error)'] = calculate_recall_per_edge(object_predict, object_target, rel_predict,
                                                                      one_hot_rel_target, input, 100)
            error = output['edge_acc_100(error)']
        output['error'] = 1 - error
        output['n_count'] = 1
        # TODO: error calculate not right (should not mean in batch)
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
    net = PointNetSG(config)
    net = net.cuda()
    net.set_params()
    # exit()
    from torchsummary import summary

    summary(net, batch_size=2, input_size=(4096, 3))
