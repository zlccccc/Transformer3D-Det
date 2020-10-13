# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from nn_distance import nn_distance, pc_nn_distance, huber_loss
from matcher3d import build_matcher
import torch
import torch.nn as nn
import numpy as np
import sys
import os

GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness
matcher = None


def compute_vote_loss(end_points, vote_xyz):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
        vote_xyz: vote xyz in cascade vote stage k

    Returns:
        vote_loss: scalar Tensor

    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1]  # B,num_seed,3
    # vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long()  # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size, num_seed, 1).repeat(1, 1, 3 * GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    # seed_gt_votes += end_points['seed_xyz'].repeat(1, 1, 3)
    seed_gt_votes = end_points['seed_xyz'].repeat(1,1,3)  # NOT VOTE

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size * num_seed, -1, 3)  # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size * num_seed, GT_VOTE_FACTOR, 3)  # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1)  # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist * seed_gt_votes_mask.float()) / (torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss


def compute_cascade_vote_loss(end_points):
    vote_stage = end_points['vote_stage']
    vote_loss = torch.zeros(1).cuda()
    if vote_stage >= 1:
        end_points['vote_loss_stage_1'] = compute_vote_loss(end_points, end_points['vote_xyz_stage_1'])
        vote_loss += end_points['vote_loss_stage_1']
    if vote_stage >= 2:
        end_points['vote_loss_stage_2'] = compute_vote_loss(end_points, end_points['vote_xyz_stage_2'])
        vote_loss += end_points['vote_loss_stage_2']
    if vote_stage >= 3:
        end_points['vote_loss_stage_3'] = compute_vote_loss(end_points, end_points['vote_xyz_stage_3'])
        vote_loss += end_points['vote_loss_stage_3']
    vote_loss /= vote_stage
    return vote_loss


def compute_bbox_loss(end_points, config, config_matcher, loss_weight_dict):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    # num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    # object_assignment = end_points['object_assignment']
    # batch_size = object_assignment.shape[0]

    metric = {}  # to do bi-matching-3d
    box_label_mask = end_points['box_label_mask']
    sizes = box_label_mask.int().sum(dim=-1)
    box_label_mask = (box_label_mask == 1)
    B, MAXN = box_label_mask.shape

    # Compute center loss (already changed)
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:,:, 0:3]
    center_dist = []
    for bs in range(B):
        batch_pred = pred_center[bs:bs+1, :, :]
        batch_gt = gt_center[bs:bs+1, box_label_mask[bs], :]
        # TODO CHECK NN DISTANCE(NOW IT IS WRONG)
        nn_dist = pc_nn_distance(batch_pred, batch_gt)[0]
        center_dist.append(nn_dist)
        # print(batch_pred.shape, gt_pred.shape, nn_dist.shape, '<< query [center] bs %d shape'%bs)
    metric['center_dist_loss'] = center_dist
    # dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center)  # dist1: BxK, dist2: BxK2
    # objectness_label = end_points['objectness_label'].float()
    # centroid_reg_loss1 = \
    #     torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    # centroid_reg_loss2 = \
    #     torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    # center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_loss, heading_residual_loss = [], []
    heading_class_pred, heading_class_gt = end_points['heading_scores'], end_points['heading_class_label']
    heading_residual_pred, heading_residual_gt = end_points['heading_residuals_normalized'], end_points['heading_residual_label'] / (np.pi/num_heading_bin) # normalized
    _, MAXQ, _ = heading_class_pred.shape

    # print(heading_class_pred.shape, heading_class_gt.shape, heading_residual_pred.shape, heading_residual_gt.shape, '<< heading class label')
    # print(heading_class_gt, heading_residual_gt)
    for bs in range(B):
        # heading_class_loss
        batch_gt = heading_class_gt[bs, None, box_label_mask[bs]].repeat(MAXQ, 1)
        batch_gt_class = batch_gt
        GT_COUNT = batch_gt.shape[1]
        if GT_COUNT == 0:  # append zero
            heading_class_loss.append(0)
            heading_residual_loss.append(0)
            continue
        batch_pred = heading_class_pred[bs, :, None, :].repeat(1, GT_COUNT, 1)
        criterion = nn.CrossEntropyLoss(reduction='none')
        batch_class_loss = criterion(batch_pred.reshape(MAXQ*GT_COUNT, num_heading_bin), batch_gt.reshape(MAXQ*GT_COUNT))
        batch_class_loss = batch_class_loss.reshape(MAXQ, GT_COUNT)
        heading_class_loss.append(batch_class_loss)
        # print(batch_pred.shape, batch_gt.shape, batch_class_loss.shape, '<< query [heading] bs %d shape'%bs)
        # heading_residual_loss
        batch_gt = heading_residual_gt[bs, None, box_label_mask[bs]].repeat(MAXQ, 1)
        GT_COUNT = batch_gt.shape[1]
        batch_pred = heading_residual_pred[bs, :, None, :].repeat(1, GT_COUNT, 1)
        batch_pred = torch.gather(batch_pred, 2, batch_gt_class.unsqueeze(-1)).squeeze(-1)
        batch_residual_loss = huber_loss(batch_pred - batch_gt, delta=1.0)
        heading_residual_loss.append(batch_residual_loss)
        # print(batch_pred.shape, batch_gt.shape, batch_gt_class.shape, batch_residual_loss.shape, '<< query [heading residual] bs %d shape'%bs)
    metric['heading_class_loss'] = heading_class_loss
    metric['heading_residual_loss'] = heading_residual_loss

    # heading_class_label = torch.gather(end_points['heading_class_label'], 1, object_assignment)  # select (B,K) from (B,K2)
    # criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    # heading_class_loss = criterion_heading_class(end_points['heading_scores'].transpose(2, 1), heading_class_label)  # (B,K)
    # heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    # heading_residual_label = torch.gather(end_points['heading_residual_label'], 1, object_assignment)  # select (B,K) from (B,K2)
    # heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    # heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    # heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
    # heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0)  # (B,K)
    # heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_loss, size_residual_loss = [], []
    size_class_pred, size_class_gt = end_points['size_scores'], end_points['size_class_label']
    size_residual_pred, size_residual_gt = end_points['size_residuals_normalized'], end_points['size_residual_label'] # normalized
    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).type_as(size_class_pred).unsqueeze(0).unsqueeze(0)  # (num_size_cluster, 3)
    _, MAXQ, _ = size_class_pred.shape

    # print(size_class_pred.shape, size_class_gt.shape, size_residual_pred.shape, size_residual_gt.shape, '<< size class label; pred and gt shape')
    # print(size_class_gt, size_residual_gt)
    for bs in range(B):
        # size_class_loss
        batch_gt = size_class_gt[bs, None, box_label_mask[bs]].repeat(MAXQ, 1)
        batch_gt_class = batch_gt  # MAXQ, GT_COUNT
        GT_COUNT = batch_gt.shape[1]
        if GT_COUNT == 0:  # append zero
            size_class_loss.append(0)
            size_residual_loss.append(0)
            continue
        batch_pred = size_class_pred[bs, :, None, :].repeat(1, GT_COUNT, 1)
        criterion = nn.CrossEntropyLoss(reduction='none')
        batch_class_loss = criterion(batch_pred.reshape(MAXQ*GT_COUNT, num_class), batch_gt.reshape(MAXQ*GT_COUNT))
        batch_class_loss = batch_class_loss.reshape(MAXQ, GT_COUNT)
        size_class_loss.append(batch_class_loss)
        # print(batch_pred.shape, batch_gt.shape, batch_class_loss.shape, '<< query [size] bs %d shape'%bs)
        # size_residual_loss
        batch_gt = size_residual_gt[bs, None, box_label_mask[bs], :].repeat(MAXQ, 1, 1) # MAXQ*GT_COUNT*C
        _, _, C = batch_gt.shape
        meansize_gt = mean_size_arr_expanded.repeat(MAXQ, GT_COUNT, 1, 1)
        # print(meansize_gt.shape, ' << meansize gt') # I think it is rignt
        meansize_gt = torch.gather(meansize_gt, 2, batch_gt_class[:, :, None, None].repeat(1, 1, 1, C)).squeeze(2)
        batch_gt /= meansize_gt
        # print(meansize_gt.shape, ' << fin meansize gt')
        _, GT_COUNT, C = batch_gt.shape
        batch_pred = size_residual_pred[bs, :, None, :, :].repeat(1, GT_COUNT, 1, 1)
        batch_gt_class_inds = batch_gt_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, C)
        batch_pred = torch.gather(batch_pred, 2, batch_gt_class_inds).squeeze(2)
        batch_residual_loss = torch.mean(huber_loss(batch_pred - batch_gt, delta=1.0), -1)
        # print(batch_pred.shape, batch_gt.shape, batch_gt_class_inds.shape, batch_residual_loss.shape, '<< query [size residual] bs %d shape'%bs)
        size_residual_loss.append(batch_residual_loss)
    metric['size_class_loss'] = size_class_loss
    metric['size_residual_loss'] = size_residual_loss
    # size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment)  # select (B,K) from (B,K2)
    # criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    # size_class_loss = criterion_size_class(end_points['size_scores'].transpose(2, 1), size_class_label)  # (B,K)
    # size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    # size_residual_label = torch.gather(end_points['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)
    # size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    # size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1)  # src==1 so it's *one-hot* (B,K,num_size_cluster)
    # size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B,K,num_size_cluster,3)
    # predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized']*size_label_one_hot_tiled, 2)  # (B,K,3)

    # mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)  # (1,1,num_size_cluster,3)
    # mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
    # size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)
    # size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized -
    #                                                       size_residual_label_normalized, delta=1.0), -1)  # (B,K,3) -> (B,K)
    # size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # matching
    # for key, value in metric.items():
    #     print([mat.shape for mat in value], key, '<<< metric items;')
    global matcher
    if matcher is None:
        matcher = build_matcher(config_matcher)
    indices, cost_matrices = matcher(metric)
    # print(indices, [c.shape for c in cost_matrices], 'cost matrices shape')
    loss = 0
    for key, batch_value in metric.items():
        assert key in loss_weight_dict.keys(), 'cost_dict should have weight'
        real_value = [value * loss_weight_dict[key] for value in batch_value]
        batch_loss = 0
        for i, val in enumerate(real_value):
            if len(indices[i][0]) == 0:  # length maybe zero
                continue
            val_loss = val[indices[i]]
            batch_loss += val_loss.mean()
        batch_loss /= B
        end_points[key] = batch_loss
        loss += batch_loss

    # 3.4 Semantic cls loss; LATER
    # Compute center loss (already changed)
    pred_sem_cls_label = end_points['sem_cls_scores']
    pred_obj_label = end_points['objectness_scores']
    gt_sem_cls_label = end_points['sem_cls_label']
    # print(gt_sem_cls_label.shape, pred_sem_cls_label.shape, '<<< sem cls label shape')
    # print(gt_sem_cls_label)
    cls_loss, obj_loss = 0, 0
    for bs in range(B):
        ind_qry, ind_obj = indices[bs]
        batch_class_gt = gt_sem_cls_label[bs, box_label_mask[bs]]
        batch_class_pred = pred_sem_cls_label[bs, :, :]
        # print(batch_class_pred.shape, batch_class_gt.shape, ind_qry, ind_obj, batch_class_gt, '<< pred,gt batch indice [%d]'%bs)
        batch_class_gt = batch_class_gt[ind_obj]
        batch_class_pred = batch_class_pred[ind_qry, :]
        if len(ind_qry) == 0:
            print('WARNING! There is a scan without any object!', flush=True)
            batch_cls_loss = 0
        else:
            criterion = nn.CrossEntropyLoss(reduction='none')
            batch_cls_loss = criterion(batch_class_pred, batch_class_gt) 
            batch_cls_loss = batch_cls_loss.mean()
        # print(batch_class_pred.shape, batch_class_gt.shape, ind_qry, ind_obj, batch_class_gt, batch_cls_loss, '<< pred,gt batch indice [%d]'%bs)
        # if have object
        batch_obj_pred = pred_obj_label[bs, :, :]
        K, _ = batch_obj_pred.shape
        batch_objectness_label = torch.zeros((K), dtype=torch.long).to(batch_obj_pred.device)
        batch_objectness_label[ind_qry] = 1
        # print(ind_qry, batch_objectness_label, '<< mask')
        criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
        batch_obj_loss = criterion(batch_obj_pred, batch_objectness_label)
        batch_obj_loss = batch_obj_loss.mean()

        cls_loss += batch_cls_loss
        obj_loss += batch_obj_loss
    # exit()
    cls_loss = cls_loss / B * loss_weight_dict['cls_loss']
    obj_loss = obj_loss / B * loss_weight_dict['obj_loss']
    end_points['cls_loss'] = cls_loss
    end_points['obj_loss'] = obj_loss
    loss = loss + cls_loss + obj_loss

    # sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
    # criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    # sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)
    # sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return loss, end_points, indices


def get_loss(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss = compute_cascade_vote_loss(end_points)
    # end_points['vote_loss'] = vote_loss * 0.2
    end_points['vote_loss'] = vote_loss * 0

    # Obj loss
    # objectness_loss, objectness_label, objectness_mask, object_assignment = \
    #     compute_objectness_loss(end_points)
    # end_points['objectness_loss'] = objectness_loss
    # end_points['objectness_label'] = objectness_label
    # end_points['objectness_mask'] = objectness_mask
    # end_points['object_assignment'] = object_assignment
    # total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    # end_points['pos_ratio'] = \
    #     torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    # end_points['neg_ratio'] = \
    #     torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    config_matcher = {
        'center_dist_loss': 20,
        'heading_class_loss': 1,
        'heading_residual_loss': 10,
        'size_class_loss': 1,
        'size_residual_loss': 10,
    }
    loss_weight_dict = config_matcher.copy()
    loss_weight_dict['cls_loss'] = 1  # for obj class (transformer)
    loss_weight_dict['obj_loss'] = 5  # for obj if used (transformer)
    bbox_loss, end_points, indices = compute_bbox_loss(end_points, config, config_matcher, loss_weight_dict)

    end_points['box_loss'] = bbox_loss

    # Final loss function
    loss = vote_loss + bbox_loss

    # --------------------------------------------
    # Some other statistics
    # obj_pred_val = torch.argmax(end_points['objectness_scores'], 2)  # B,K
    # obj_acc = torch.sum((obj_pred_val == objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    # end_points['obj_acc'] = obj_acc

    return loss, end_points
