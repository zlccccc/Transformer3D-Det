# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
from data_viz_helper import save_obj

DUMP_CONF_THRESH = 0.5 # Dump boxes with obj prob larger than that.

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

idx_beg, save_count = 0, 0
def dump_results(end_points, dump_dir, config, inference_switch=False):
    global idx_beg, save_count
    if True:
        mAP_now = list(end_points['mAP'])
        mAP_mean = [item for item in mAP_now if not np.isnan(item)]
        mAP_mean = np.mean(mAP_mean)
        AR_now = list(end_points['AR'])
        AR_mean = [item for item in AR_now if not np.isnan(item)]
        AR_mean = np.mean(AR_mean)
        map_val_str = '%.2f_P%.2f_R%.2f' %(mAP_mean * AR_mean, mAP_mean, AR_mean)
        idx_beg += 1
        # if mAP_mean * AR_mean > 0.7 or True:
        # if mAP_mean * AR_mean > 0.9:# or True:
        if '0461_00' in end_points['scan_name'][0] or '0196_00' in end_points['scan_name'][0]:# or True:
            save_count += 1
            print('result saved idx count', save_count, '/', idx_beg, \
                                            mAP_mean * AR_mean, mAP_mean, AR_mean, flush=True)
        else:
            print('DONT SAVE THIS!', end_points['scan_name'], idx_beg, mAP_mean * AR_mean, mAP_mean, AR_mean, flush=True)
            return
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''
    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = end_points['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]
    assert batch_size == 1, 'batch size should be 1'

    # LABELS
    gt_center = end_points['center_label'].cpu().numpy() # (B,MAX_NUM_OBJ,3)
    gt_mask = end_points['box_label_mask'].cpu().numpy() # B,K2
    gt_heading_class = end_points['heading_class_label'].cpu().numpy() # B,K2
    gt_heading_residual = end_points['heading_residual_label'].cpu().numpy() # B,K2
    gt_size_class = end_points['size_class_label'].cpu().numpy() # B,K2
    gt_size_residual = end_points['size_residual_label'].cpu().numpy() # B,K2,3
    objectness_label = end_points['objectness_label'].detach().cpu().numpy() # (B,K,)
    objectness_mask = end_points['objectness_mask'].detach().cpu().numpy() # (B,K,)

    # NETWORK OUTPUTS
    seed_xyz = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    if 'vote_xyz' in end_points:
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
        vote_xyz = end_points['vote_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    objectness_scores = end_points['objectness_scores'].detach().cpu().numpy() # (B,K,2)
    pred_center = end_points['center'].detach().cpu().numpy() # (B,K,3)
    pred_heading_class = torch.argmax(end_points['heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(end_points['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
    pred_size_class = torch.argmax(end_points['size_scores'], -1) # B,num_proposal
    pred_size_residual = torch.gather(end_points['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

    pred_sem_cls = torch.argmax(end_points['sem_cls_scores'], dim=-1).cpu().numpy()
    # OTHERS
    pred_mask = end_points['pred_mask'] # B,num_proposal
    base_dir = dump_dir
    # idx_beg = 0

    for i in range(batch_size):
        scan_id = int(end_points['scan_name'][i][5:9])
        print(end_points['scan_name'], i+idx_beg, '<< scan idx', scan_id, len(list(end_points['mAP'])), '<< mAP', flush=True)

        dump_dir = os.path.join(base_dir, '%s_[%s]'%(map_val_str,end_points['scan_name'][i]))
        if not os.path.exists(dump_dir):
            os.system('mkdir %s'%(dump_dir))
        gt_obj_number = save_obj('/mnt/lustre/liujie4/big/scannet_train_detection_data', end_points['scan_name'][i], dump_dir)

        pc = point_clouds[i,:,:]
        objectness_prob = softmax(objectness_scores[i,:,:])[:,1] # (K,)

        pred_obj_number = np.sum(objectness_prob>DUMP_CONF_THRESH)
        print(gt_obj_number, pred_obj_number, ' << gt and pred bbox number', flush=True)
        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, '%06d_pc.ply'%(scan_id)))
        pc_util.write_ply(seed_xyz[i,:,:], os.path.join(dump_dir, '%06d_seed_pc.ply'%(scan_id)))
        if 'vote_xyz' in end_points:
            pc_util.write_ply(end_points['vote_xyz'][i,:,:], os.path.join(dump_dir, '%06d_vgen_pc.ply'%(scan_id)))
            pc_util.write_lines_as_cylinders(np.stack([end_points['vote_xyz'][i,:,:].detach().cpu().numpy(),seed_xyz[i,:,:]],axis=1), os.path.join(dump_dir, '%06d_line_seed_vote.ply'%(scan_id)))
            pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(scan_id)))
            aggregated_vote_inds = end_points['aggregated_vote_inds']  # TODO
            aggregated_seed_xyz = torch.gather(end_points['seed_xyz'], 1, aggregated_vote_inds[...,None].long().repeat(1,1,3))
            aggregated_seed_xyz = aggregated_seed_xyz.detach().cpu().numpy()
            pc_util.write_ply(aggregated_seed_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_seed_pc.ply'%(scan_id)))

        pc_util.write_ply(pred_center[i,:,0:3], os.path.join(dump_dir, '%06d_proposal_pc.ply'%(scan_id)))

        if 'transformer_weighted_xyz_all' in end_points:
            # print(end_points['transformer_weighted_xyz_all'].shape, '<< shape all', flush=True)
            transformer_xyz = end_points['transformer_weighted_xyz_all'].detach().cpu().numpy()
            for ly in range(len(transformer_xyz)):
                pc_util.write_ply(transformer_xyz[ly,i,:,:], os.path.join(dump_dir, '%06d_transformer_%d_pc.ply'%(scan_id, ly)))
                pc_util.write_lines_as_cylinders(np.stack([transformer_xyz[ly,i,:,0:3],aggregated_vote_xyz[i,:,0:3]],axis=1), os.path.join(dump_dir, '%06d_transformer_%d_line_vote_transformer.ply'%(scan_id,ly)))
                if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
                    pc_util.write_ply(transformer_xyz[ly,i,objectness_prob>DUMP_CONF_THRESH,0:3], os.path.join(dump_dir, '%06d_transformer_%d_confident_pc.ply'%(scan_id,ly)))
                    pc_util.write_lines_as_cylinders(np.stack([transformer_xyz[ly,i,objectness_prob>DUMP_CONF_THRESH,0:3],aggregated_vote_xyz[i,objectness_prob>DUMP_CONF_THRESH,0:3]],axis=1), os.path.join(dump_dir, '%06d_transformer_%d_confident_line_vote_transformer.ply'%(scan_id,ly)))

        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            confident_vote = aggregated_vote_xyz[i,objectness_prob>DUMP_CONF_THRESH,0:3]
            confident_seed = aggregated_seed_xyz[i,objectness_prob>DUMP_CONF_THRESH,0:3]
            confident_prop = pred_center[i,objectness_prob>DUMP_CONF_THRESH,0:3]
            # pc_util.write_ply(aggregated_vote_xyz[i,objectness_prob>DUMP_CONF_THRESH,0:3], os.path.join(dump_dir, '%06d_confident_aggregated_vote_xyz.ply'%(scan_id)))
            # pc_util.write_ply(aggregated_seed_xyz[i,objectness_prob>DUMP_CONF_THRESH,0:3], os.path.join(dump_dir, '%06d_confident_aggregated_seed_xyz.ply'%(scan_id)))
            # pc_util.write_ply(pred_center[i,objectness_prob>DUMP_CONF_THRESH,0:3], os.path.join(dump_dir, '%06d_confident_proposal_pc.ply'%(scan_id)))
            pc_util.write_ply(confident_vote, os.path.join(dump_dir, '%06d_confident_aggregated_vote_xyz.ply'%(scan_id)))
            pc_util.write_ply(confident_seed, os.path.join(dump_dir, '%06d_confident_aggregated_seed_xyz.ply'%(scan_id)))
            pc_util.write_ply(confident_prop, os.path.join(dump_dir, '%06d_confident_proposal_pc.ply'%(scan_id)))

            pc_util.write_lines_as_cylinders(np.stack([confident_vote,confident_seed],axis=1), os.path.join(dump_dir, '%06d_confident_line_seed_vote.ply'%(scan_id)))
            pc_util.write_lines_as_cylinders(np.stack([confident_vote,confident_prop],axis=1), os.path.join(dump_dir, '%06d_confident_line_vote_prop.ply'%(scan_id)))

        # Dump predicted bounding boxes
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            num_proposal = pred_center.shape[1]
            obbs = []
            for j in range(num_proposal):
                obb = config.param2obb(pred_center[i,j,0:3], pred_heading_class[i,j], pred_heading_residual[i,j],
                                pred_size_class[i,j], pred_size_residual[i,j])
                obbs.append(obb)
            if len(obbs)>0:
                case_dump_dir = os.path.join(dump_dir, 'cls_obbs')
                if not os.path.exists(case_dump_dir):
                    os.mkdir(case_dump_dir)
                obbs = np.vstack(tuple(obbs)) # (num_proposal, 7)
                pc_util.write_oriented_bbox(obbs[objectness_prob>DUMP_CONF_THRESH,:], os.path.join(dump_dir, '%06d_pred_confident_bbox.ply'%(scan_id)))
                confident_nms_mask = np.logical_and(objectness_prob>DUMP_CONF_THRESH, pred_mask[i,:]==1)
                pred_obj_nms_number= np.sum(confident_nms_mask)
                pc_util.write_oriented_bbox(obbs[confident_nms_mask,:], os.path.join(dump_dir, '%06d_pred_confident_nms_bbox.ply'%(scan_id)))
                pc_util.write_oriented_bbox(obbs[confident_nms_mask,:], os.path.join(case_dump_dir, '%06d_pred_confident_nms_bbox.ply'%(scan_id)))
                pc_util.write_oriented_bbox(obbs[pred_mask[i,:]==1,:], os.path.join(dump_dir, '%06d_pred_nms_bbox.ply'%(scan_id)))
                pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%06d_pred_bbox.ply'%(scan_id)))
                sem_cls = pred_sem_cls[i]
                # print(obbs.shape, sem_cls, '<< obbs shape')
                for l in np.unique(sem_cls):
                    sem_cls_mask = (sem_cls == l)
                    sem_conf_mask = np.logical_and(sem_cls_mask, confident_nms_mask)
                    if np.sum(sem_conf_mask) > 0:
                        pc_util.write_oriented_bbox(obbs[sem_conf_mask], os.path.join(case_dump_dir, '%06d_%02d_pred_confident_nms_bbos.ply'%(scan_id,l)))

        # Return if it is at inference time. No dumping of groundtruths
        if inference_switch:
            assert False
            continue

        # dump_dir = os.path.join(base_dir, '%06d'%(scan_id))
        if np.sum(objectness_label[i,:])>0:
            pc_util.write_ply(pred_center[i,objectness_label[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_positive_proposal_pc.ply'%(scan_id)))
        if np.sum(objectness_mask[i,:])>0:
            pc_util.write_ply(pred_center[i,objectness_mask[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_mask_proposal_pc.ply'%(scan_id)))
        pc_util.write_ply(gt_center[i,:,0:3], os.path.join(dump_dir, '%06d_gt_centroid_pc.ply'%(scan_id)))
        pc_util.write_ply_color(pred_center[i,:,0:3], objectness_label[i,:], os.path.join(dump_dir, '%06d_proposal_pc_objectness_label.obj'%(scan_id)))

        # Dump GT bounding boxes
        obbs = []
        for j in range(gt_center.shape[1]):
            if gt_mask[i,j] == 0: continue
            obb = config.param2obb(gt_center[i,j,0:3], gt_heading_class[i,j], gt_heading_residual[i,j],
                            gt_size_class[i,j], gt_size_residual[i,j])
            obbs.append(obb)
        if len(obbs)>0:
            obbs = np.vstack(tuple(obbs)) # (num_gt_objects, 7)
            pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%06d_gt_bbox.ply'%(scan_id)))

        # OPTIONALL, also dump prediction and gt details
        if 'batch_pred_map_cls' in end_points:
            fout = open(os.path.join(dump_dir, '%06d_pred_map_cls.txt'%(i)), 'w')
            for t in end_points['batch_pred_map_cls'][i]:
                fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write(' '+str(t[2]))
                fout.write('\n')
            fout.close()
        if 'batch_gt_map_cls' in end_points:
            fout = open(os.path.join(dump_dir, '%06d_gt_map_cls.txt'%(i)), 'w')
            for t in end_points['batch_gt_map_cls'][i]:
                fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write('\n')
            fout.close()

        percentage = min(pred_obj_nms_number/(gt_obj_number+0.00001), gt_obj_number/(pred_obj_nms_number+0.00001))
        print('Pred Number %d:%d Real'%(pred_obj_nms_number, gt_obj_number), flush=True)
        newname = os.path.join(base_dir, 'Percent_%.3f_%s_[%s]'%(percentage,map_val_str,end_points['scan_name'][i]) + '_pred_%d_real_%d_allnonms_%d' % (pred_obj_nms_number, gt_obj_number, pred_obj_number))
        # newname = dump_dir + 'Percent_%.3f_' %percentage + '_pred_%d_real_%d_allnonms_%d' % (pred_obj_nms_number, gt_obj_number, pred_obj_number)
        print('move to', newname, flush=True)
        os.rename(dump_dir, newname)
