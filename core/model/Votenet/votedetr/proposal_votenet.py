# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
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

from pointnet2_modules import PointnetSAModuleVotes
from nn_distance import huber_loss
import pointnet2_utils
from detr3d import DETR3D


def decode_scores(output_dict, end_points,  num_class, num_heading_bin, num_size_cluster, mean_size_arr, center_with_bias=False, quality_channel=False):  # TODO CHANGE IT
    # net_transposed = net.transpose(2, 1)
    pred_logits = output_dict['pred_logits']
    pred_boxes = output_dict['pred_boxes']
    # print(pred_boxes, flush=True) # <<< ??? pred box outputs
    # print(pred_logits.shape, pred_boxes.shape, '<< decode shape score')

    batch_size = pred_boxes.shape[0]
    num_proposal = pred_boxes.shape[1]

    assert pred_logits.shape[-1] == 2+num_class, 'pred_logits.shape wrong'
    objectness_scores = pred_logits[:,:,0:2]  # TODO CHANGE IT; JUST SOFTMAXd
    end_points['objectness_scores'] = objectness_scores
    sem_cls_scores = pred_logits[:,:,2:2+num_class] # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores

    bbox_args_shape = 3+num_heading_bin*2+num_size_cluster*4
    if quality_channel:
        bbox_args_shape += 1
    assert pred_boxes.shape[-1] == bbox_args_shape, 'pred_boxes.shape wrong'
    center = pred_boxes[:,:,0:3] # (batch_size, num_proposal, 3) TODO RESIDUAL
    if center_with_bias:
        # print('CENTER ADDING VOTE-XYZ', flush=True)
        base_xyz = end_points['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
        # print('Using Center With Bias', output_dict.keys())
        if 'transformer_weighted_xyz' in output_dict.keys():
            end_points['transformer_weighted_xyz_all'] = output_dict['transformer_weighted_xyz_all']  # just for visualization
            transformer_xyz = output_dict['transformer_weighted_xyz']
            # print(transformer_xyz[0, :4], base_xyz[0, :4], 'from vote helper', flush=True)
            # print(center.shape, transformer_xyz.shape)
            center = center + transformer_xyz
        else:
            center = center + base_xyz  # residual
    else:
        raise NotImplementedError('center without bias(for decoder): not Implemented')
    end_points['center'] = center

    heading_scores = pred_boxes[:,:,3:3+num_heading_bin]  # theta; todo change it
    heading_residuals_normalized = pred_boxes[:,:,3+num_heading_bin:3+num_heading_bin*2]
    end_points['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

    size_scores = pred_boxes[:,:,3+num_heading_bin*2:3+num_heading_bin*2+num_size_cluster]
    size_residuals_normalized = pred_boxes[:,:,3+num_heading_bin*2+num_size_cluster:3+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3 TODO REMOVE BBOX-SIZE-DEFINED
    # size_residuals_normalized = size_residuals_normalized.sigmoid() * 2 - 1
    # size_residuals_normalized = size_residuals_normalized.atan() / math.pi  # -0.5 to 0.5
    # print('size normalized value max and min', size_residuals_normalized.max(), size_residuals_normalized.min(), size_residuals_normalized.std(), flush=True)
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    mean_size = torch.from_numpy(mean_size_arr.astype(np.float32)).type_as(pred_boxes).unsqueeze(0).unsqueeze(0)
    end_points['size_residuals'] = size_residuals_normalized * mean_size
    # print(3+num_heading_bin*2+num_size_cluster*4, ' <<< bbox heading and size tensor shape')
    if quality_channel:
        end_points['quality_weight'] = pred_boxes[:,:,3+num_heading_bin*2+num_size_cluster*4] 

    return end_points


def farthest_point_sample_with_weight(xyz, npoint, weight=None, already_chosen=None):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        if already_chosen is not None and i < already_chosen.shape[-1]:  # assign label
            farthest = already_chosen[:, i].long()
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        if weight is not None:
            dist = dist * weight
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        # print(dist, farthest)
        # print(farthest, '<< farthest dist')
    return centroids


class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling,
                 seed_feat_dim=256, config_transformer=None, quality_channel=False):
        super().__init__()
        print(config_transformer, '<< config transformer ')
        if 'voting_transformer' in config_transformer:
            voting_transformer = config_transformer.voting_transformer
            self.voting_transformer_config = voting_transformer
            self.voting_transformer = DETR3D(voting_transformer, input_channels=seed_feat_dim, class_output_shape=0, bbox_output_shape=256)  # patch-move
            seed_feat_dim = 256
        else:
            self.voting_transformer = None

        transformer_type = config_transformer.get('transformer_type', 'enc_dec')
        position_type = config_transformer.get('position_type', 'vote')
        self.transformer_type = transformer_type
        self.position_type = position_type
        self.center_with_bias = 'dec' not in transformer_type

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.quality_channel = quality_channel

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
        self.detr = DETR3D(config_transformer, input_channels=128, class_output_shape=2+num_class, bbox_output_shape=3+num_heading_bin*2+num_size_cluster*4+int(quality_channel))

    def forward(self, initial_xyz, xyz, features, end_points):  # initial_xyz and xyz(voted): just for encoding
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        # print(initial_xyz, end_points['seed_xyz'], '<< TESTING', flush=True)
        if self.voting_transformer is not None:
            # print(xyz[:, :2, :], initial_xyz[:, :2, :], '<< input xyz', flush=True)
            output_dict = self.voting_transformer(xyz, features.permute(0, 2, 1), end_points)
            assert 'transformer_weighted_xyz' in output_dict.keys(), 'you should use transformer to move point position'
            cascade_xyz = output_dict['transformer_weighted_xyz_all']  # todo may add weight
            xyz = output_dict['transformer_weighted_xyz'].contiguous()
            if self.voting_transformer_config.get('add', False):
                # print('feature adding')
                features = features + output_dict['pred_boxes'].permute(0, 2, 1).contiguous()
            else:
                features = output_dict['pred_boxes'].permute(0, 2, 1).contiguous()
            # print(xyz[:, :3], initial_xyz[:, :3], '<< xyz', flush=True)
            # voting fin value
            end_points['vote_xyz'] = xyz
            end_points['vote_features'] = features
            end_points['vote_stage'] = 1
            end_points['vote_xyz_stage_1'] = xyz
            # xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            # sample_inds = fps_inds
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps':  # regard initial_xyz as seed
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            # print('shape', self.num_proposal, initial_xyz.shape)
            sample_inds = pointnet2_utils.furthest_point_sample(initial_xyz, self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = initial_xyz.shape[1]
            batch_size = initial_xyz.shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'vote_dist_fps':
            xyz_dist = huber_loss(xyz - initial_xyz, delta=1.0)
            xyz_dist = torch.mean(xyz_dist, dim=-1)
            # print(xyz_dist, flush=True)
            # print(initial_xyz.shape, xyz.shape, '<< shape`:')
            sorted_dist, sorted_ind = torch.sort(xyz_dist, dim=1, descending=True)
            # print(sorted_dist[:, 127], '<< sorted dist dist')
            dist_num_proposal = self.num_proposal // 2
            ini_inds = pointnet2_utils.furthest_point_sample(initial_xyz, dist_num_proposal)
            dist_weight = xyz_dist
            dist_weight[dist_weight>0.1] = 0.1
            # print(dist_weight, dist_weight.mean(dim=-1))
            sample_inds = farthest_point_sample_with_weight(initial_xyz, self.num_proposal, dist_weight, ini_inds)
            sample_inds = sample_inds.int()
            # print(ini_inds, sample_inds, '<< fps inds')
            # assert False
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            raise NotImplementedError('Unknown sampling strategy: %s. Exiting!' % (self.sampling))
        
        end_points['aggregated_vote_xyz'] = xyz  # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_features'] = features.permute(0, 2, 1).contiguous() # (batch_size, num_proposal, 128)
        end_points['aggregated_vote_inds'] = sample_inds  # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ----------  TODO PROPOSAL GENERATION AND CHANGE LOSS GENERATION
        # print(features.mean(), features.std(), ' << first,votenet forward features mean and std', flush=True) # TODO CHECK IT
        features = F.relu(self.bn1(self.conv1(features)))
        features = F.relu(self.bn2(self.conv2(features)))
        # features = self.conv3(features)  # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)
        # print(features.mean(), features.std(), ' << votenet forward features mean and std', flush=True) # TODO CHECK IT

        # end_points = decode_scores(features, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)

        # _xyz = torch.gather(initial_xyz, 1, sample_inds.long().unsqueeze(-1).repeat(1,1,3))
        # print(initial_xyz.shape, xyz.shape, sample_inds.shape, _xyz.shape, '<< sample xyz shape', flush=True)
        features = features.permute(0, 2, 1)
        # print(xyz.shape, features.shape, '<< detr input feature dim')
        # output_dict = self.detr(torch.cat([xyz, _xyz], dim=-1), features, end_points)
        if self.position_type == 'vote':
            output_dict = self.detr(xyz, features, end_points)
        else:
            # print(xyz.shape, features.shape, initial_xyz.shape, '<<< initial position shape', flush=True)
            # chosen_xyz = torch.gather(end_points['vote_xyz'], 1, sample_inds.unsqueeze(-1).repeat(1, 1, 3).long())
            # print(chosen_xyz[0, :5], xyz[0, :5])
            chosen_xyz = torch.gather(initial_xyz, 1, sample_inds.unsqueeze(-1).repeat(1, 1, 3).to(initial_xyz.device).long())
            output_dict = self.detr(chosen_xyz, features, end_points)
        # output_dict = self.detr(xyz, features, end_points)
        end_points = decode_scores(output_dict, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr, self.center_with_bias, quality_channel=self.quality_channel)

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
