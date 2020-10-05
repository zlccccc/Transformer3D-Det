import torch
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
            self.parse_predictions = parse_predictions
            self.parse_groundtruths = parse_groundtruths 
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
        for key in input:
            assert(key not in output)
            output[key] = input[key]
        loss, output = self.criterion(output, self.DATASET_CONFIG)
        fin_out = {}
        for key, value in output.items():
            # if isinstance(value, torch.Tensor):
            #     print('cal loss', key, value.shape)
            # else:
            #     print('cal loss not tensor', key, value)
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                # print('cal loss --- real loss', key, value)
                if 'loss' not in key:
                    key = key + '_loss'
                fin_out[key] = value.detach()
        fin_out['loss'] = loss
        # print(fin_out, ' <<<  fin out (output); loss calculating')
        return fin_out

    def initialize_error(self):  # TODO; IoUCalculator; must for dataset
        print('---- Initialize error calculation ----')
        self.ap_calculator_25 = self.APCalculator(ap_iou_thresh=0.25,
                                                  class2type_map=self.DATASET_CONFIG.class2type)
        self.ap_calculator_50 = self.APCalculator(ap_iou_thresh=0.50,
                                                  class2type_map=self.DATASET_CONFIG.class2type)
        self.CONFIG_DICT = {'remove_empty_box':False, 'use_3d_nms':True,
                            'nms_iou':0.25, 'use_old_type_nms':False, 'cls_nms':True,
                            'per_class_proposal': True, 'conf_thresh':0.05,
                            'dataset_config': self.DATASET_CONFIG}
        self.ap_n_count = 0

    # TODO
    def calculate_error(self, input, output):
        for key in input:
            assert(key not in output)
            output[key] = input[key]
        loss, output = self.criterion(output, self.DATASET_CONFIG)
        fin_out = {}
        for key, value in output.items():
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                # print('cal loss --- real loss', key, value)
                if 'loss' not in key:
                    key = key + '_loss'
                key = key.replace('loss', 'error')
                fin_out[key] = value.detach()
        fin_out['all_error'] = loss.detach()
        # print(fin_out, ' <<<  fin out (output); error calculating', flush=True)

        self.ap_n_count += 1
        batch_pred_map_cls = self.parse_predictions(output, self.CONFIG_DICT) 
        batch_gt_map_cls = self.parse_groundtruths(output, self.CONFIG_DICT) 
        self.ap_calculator_25.step(batch_pred_map_cls, batch_gt_map_cls)
        self.ap_calculator_50.step(batch_pred_map_cls, batch_gt_map_cls)
        # metric calculating
        fin_out['error'] = 0
        fin_out['n_count'] = 1
        # TODO: error calculate not right (should not mean in batch)
        # fin_out['n_count'] = 1
        return fin_out


    def final_error(self, input, output):  # more infomation
        metrics_dict_25 = self.ap_calculator_25.compute_metrics()
        output['mAP@0.25_error'] = metrics_dict_25['mAP'] * self.ap_n_count
        output['AR@0.25_error'] = metrics_dict_25['AR'] * self.ap_n_count
        metrics_dict_50 = self.ap_calculator_50.compute_metrics()
        output['mAP@0.50_error'] = metrics_dict_50['mAP'] * self.ap_n_count
        output['AR@0.50_error'] = metrics_dict_50['AR'] * self.ap_n_count
        output['error'] = self.ap_n_count - output['mAP@0.50_error'] 
        # Evaluate average precision
        print('Eval--mAP@0.25', metrics_dict_25['mAP'])
        print('Eval--mAP@0.50', metrics_dict_50['mAP'])
        print('Eval-----AR@0.25', metrics_dict_25['AR'])
        print('Eval-----AR@0.50', metrics_dict_50['AR'])
        for key in metrics_dict_25:
            print('eval %s: %f'%(key, metrics_dict_25[key]), flush=True)
        return output

