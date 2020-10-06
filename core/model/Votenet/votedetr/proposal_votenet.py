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
sys.path.append(os.path.join(ROOT_DIR, 'votedetr'))  # DETR

from pointnet2_modules import PointnetSAModuleVotes
import pointnet2_utils
from .detr3d import DETR3D


def decode_scores(output_dict, end_points):  # TODO CHANGE IT
    # net_transposed = net.transpose(2, 1)
    # TODO CHANGE OUTPUT_DICT
    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling,
                 seed_feat_dim=256, config_transformer=None):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        # JUST FOR
        self.detr = DETR3D(config_transformer, 128, num_class, num_class * 4, 79)

    def forward(self, initial_xyz, xyz, features, end_points):  # initial_xyz and xyz(voted): just for encoding
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps':  # regard initial_xyz as seed
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(initial_xyz, self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = initial_xyz.shape[1]
            batch_size = initial_xyz.shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            raise NotImplementedError('Unknown sampling strategy: %s. Exiting!' % (self.sampling))
        end_points['aggregated_vote_xyz'] = xyz  # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds  # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ----------  TODO PROPOSAL GENERATION AND CHANGE LOSS GENERATION
        features = F.relu(self.bn1(self.conv1(features)))
        features = F.relu(self.bn2(self.conv2(features)))
        # features = self.conv3(features)  # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        # end_points = decode_scores(features, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)

        output_dict = self.detr(xyz, features)
        end_points = decode_scores(output_dict, end_points)

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
