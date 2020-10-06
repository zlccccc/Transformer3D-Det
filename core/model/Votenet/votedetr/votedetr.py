# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
# VOTENET AND DETR MODEL CODEBASE PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
sys.path.append(MODEL_DIR)
sys.path.append(BASE_DIR)
print('\n'.join(sys.path))
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_votenet import ProposalModule
from dump_helper import dump_results
from loss_helper import get_loss


class VoteDetr(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps', vote_stage=1,
                 config_transformer=None):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.vote_stage = vote_stage
        assert vote_stage in [1, 2, 3]

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen1 = VotingModule(self.vote_factor, 256)
        self.vgen2 = VotingModule(self.vote_factor, 256)
        self.vgen3 = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
                                   mean_size_arr, num_proposal, sampling, config_transformer=config_transformer)

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}

        end_points = self.backbone_net(inputs['point_clouds'], end_points)

        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features

        if self.vote_stage >= 1:
            xyz, features = self.vgen1(xyz, features)
            end_points['vote_xyz_stage_1'] = xyz
        if self.vote_stage >= 2:
            xyz, features = self.vgen2(xyz, features)
            end_points['vote_xyz_stage_2'] = xyz
        if self.vote_stage >= 3:
            xyz, features = self.vgen3(xyz, features)
            end_points['vote_xyz_stage_3'] = xyz

        end_points['vote_stage'] = self.vote_stage
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        seed_xyz = end_points['seed_xyz']  # initial
        end_points = self.pnet(seed_xyz, xyz, features, end_points)  # for feature

        return end_points
