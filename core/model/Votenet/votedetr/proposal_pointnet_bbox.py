# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(BASE_DIR)  # DETR

from proposal_votenet import decode_scores
import pointnet2_utils
from detr3d import DETR3D


def decode_scores_bbox(output_dict, end_points,  num_class, num_heading_bin, num_size_cluster, mean_size_arr):  # TODO CHANGE IT
    # net_transposed = net.transpose(2, 1)
    # TODO CHANGE OUTPUT_DICT
    pred_logits = output_dict['pred_logits']
    pred_boxes = output_dict['pred_boxes']
    # print(pred_boxes, flush=True) # <<< ??? pred box outputs
    # print(pred_logits.shape, pred_boxes.shape, '<< decode shape score')

    batch_size = pred_boxes.shape[0]
    num_proposal = pred_boxes.shape[1]

    objectness_scores = pred_logits[:,:,0:2]  # TODO CHANGE IT; JUST SOFTMAX
    end_points['objectness_scores'] = objectness_scores
    sem_cls_scores = pred_logits[:,:,2:2+num_class] # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores

    # get house size
    point_cloud = end_points['point_clouds']
    max_position = torch.max(point_cloud, dim=1)[0]
    min_position = torch.min(point_cloud, dim=1)[0]
    point_cloud_size = max_position - min_position
    normalize_size = torch.Tensor([12, 12, 6]).type_as(point_cloud).unsqueeze(0).repeat(batch_size, 1)
    normalize_size = torch.max(normalize_size, -min_position * 2)
    normalize_size = torch.max(normalize_size, max_position * 2)
    end_points['normalize_size'] = normalize_size  # just for pc normalization
    # print('decode score box done!', max_position, min_position, point_cloud_size, normalize_size)
    # print('decode score box done! normalize', normalize_size)
    # exit()

    normalize_size = normalize_size.unsqueeze(1).repeat(1, num_proposal, 1)
    center = pred_boxes[:,:,0:3].sigmoid() # (batch_size, num_proposal, 3) TODO RESIDUAL
    center = (center - 0.5) * normalize_size
    end_points['center'] = center
    box_size = pred_boxes[:,:,3:6].sigmoid()
    box_size = box_size * normalize_size
    end_points['bbox_size'] = box_size

    heading_scores = pred_boxes[:,:,6:6+num_heading_bin]  # theta; todo change it
    heading_residuals_normalized = pred_boxes[:,:,6+num_heading_bin:6+num_heading_bin*2]
    end_points['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin


    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal,
                 seed_feat_dim=256, config_transformer=None):
        super().__init__()
        print(config_transformer, '<< config transformer ')

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.seed_feat_dim = seed_feat_dim

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(self.seed_feat_dim, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        # JUST FOR
        self.detr = DETR3D(config_transformer, input_channels=128, class_output_shape=2+num_class, bbox_output_shape=3+3+num_heading_bin*2)

    def forward(self, xyz, features, end_points):  # initial_xyz and xyz(voted): just for encoding
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        # --------- PROPOSAL GENERATION ----------
        # print(features.mean(), features.std(), ' << first,votenet forward features mean and std', flush=True) # TODO CHECK IT
        features = F.relu(self.bn1(self.conv1(features)))
        features = F.relu(self.bn2(self.conv2(features)))
        # print(features.mean(), features.std(), ' << votenet forward features mean and std', flush=True) # TODO CHECK IT

        # end_points = decode_scores(features, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)

        # print(initial_xyz.shape, xyz.shape, features.shape, '<< pointnet down xyz shape', flush=True)
        features = features.permute(0, 2, 1)
        # print(xyz.shape, features.shape, '<< detr input feature dim')
        output_dict = self.detr(xyz, features, end_points)
        # print('??  output dict done')
        end_points = decode_scores_bbox(output_dict, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)

        return end_points


if __name__ == "__main__":
    pass
    # from easydict import EasyDict
    # from model_util_scannet import ScannetDatasetConfig
    # DATASET_CONFIG = ScannetDatasetConfig()
    # config = {
    #     'num_target': 10,
    # }
    # config = EasyDict(config)
    # model = ScannetDatasetConfig(num_class=DATASET_CONFIG.num_class,
    #                              num_heading_bin=DATASET_CONFIG.num_heading_bin,
    #                              num_size_cluster=DATASET_CONFIG.num_size_cluster,
    #                              mean_size_arr=DATASET_CONFIG.mean_size_arr,
    #                              num_proposal=config.num_target,
    #                              sampling="vote_fps")
    # initial_xyz = torch.randn(3, 128, 3)
    # xyz = torch.randn(3, 128, 3)
    # features = torch.randn(3, 128, 128)
    # end_points = model(initial_xyz, xyz, features, {})
    # print(end_points)
