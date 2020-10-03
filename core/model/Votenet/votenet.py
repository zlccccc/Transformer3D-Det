import torch.nn as nn
from core.model.task_basemodel.backbone.base_model import base_module

# def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
#              input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps', vote_stage=1):


class votenet(base_module):
    def __init__(self, config):
        super(votenet, self).__init__()
        if config.task_type == 'Scannetv2':
            from model_util_scannet import ScannetDatasetConfig
            DATASET_CONFIG = ScannetDatasetConfig()
            self.DATASET_CONFIG = DATASET_CONFIG
        else:
            raise NotImplementedError(config.task_type)
        if config.net_type == 'votenet':
            from .models.votenet import VoteNet, get_loss
            from ap_helper import APCalculator, parse_predictions, parse_groundtruths
            self.APCalculator = APCalculator
            self.net = VoteNet(num_class=DATASET_CONFIG.num_class,
                               num_heading_bin=DATASET_CONFIG.num_heading_bin,
                               num_size_cluster=DATASET_CONFIG.num_size_cluster,
                               mean_size_arr=DATASET_CONFIG.mean_size_arr,
                               num_proposal=config.num_target,
                               input_feature_dim=config.num_input_channel,
                               vote_factor=config.vote_factor,
                               sampling=config.cluster_sampling)
            self.criterion = get_loss
        else:
            raise NotImplementedError(config.net_type)
        self.init_relu = 'relu'
        self.init_params(nn.BatchNorm2d, init_type='kaiming_normal')

    def _forward(self, input):
        return self.net(input)

    def calculate_loss(self, input, output):
        loss = 0
        for key in input:
            assert(key not in output)
            output[key] = input[key]
        loss, output = self.criterion(output, self.DATASET_CONFIG)
        fin_out = {}
        for key, value in output:
            print(key, value)
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                fin_out[key] = value.detach()
        fin_out['loss'] = loss
        print(fin_out, ' <<<  fin out (output); loss calculating')
        return fin_out

    def initialize_error(self):  # TODO; IoUCalculator; must for dataset
        print('---- Initialize error calculation ----')
        self.ap_calculator_25 = self.APCalculator(ap_iou_thresh=0.25,
                                                  class2type_map=self.DATASET_CONFIG.class2type)
        self.ap_calculator_50 = self.APCalculator(ap_iou_thresh=0.50,
                                                  class2type_map=self.DATASET_CONFIG.class2type)
        self.iou_n_count = 1

    # TODO
    def calculate_error(self, input, output):
        error = 1
        self.ap_calculator_25
        output['error'] = 1 - error
        output['n_count'] = 1
        # TODO: error calculate not right (should not mean in batch)
        return output



    def calculate_error(self, input, output):
        # print(self.mode, flush=True)
        output['labels'] = input['labels']
        output = reduce_points(output, self.config)
        acc, output = compute_acc(output)
        output['acc(error)'] = output['acc']
        output['acc(error_count)'] = 1
        output['error'] = 1 - output['acc(error)']
        if self.calculator is not None:  # already last
            self.calculator.add_data(output)
            mean_iou, iou_list = self.calculator.compute_iou()
            iou_list = np.array(iou_list)
            # print(mean_iou, iou_list)
            output['mean_iou(error)(mean)'] = mean_iou
            output['iou_list(error)(mean)'] = iou_list
            self.iou_n_count = 0
        output['n_count'] = 1
        return output


if __name__ == "__main__":
    import sys
    import os
    from easydict import EasyDict
