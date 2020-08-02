import torch
import numpy as np
import torch.nn.functional as F


def split_rel(input):
    sub_rel_points = input['rel_point_set']
    sub_one_hot_rel_target = input['rel_cls_one_hot']
    sub_rel_idx = input['rel_idx']
    sub_rel_mask = input['rel_mask']
    multi_batch_rel_mask = input['multi_batch_rel_mask']
    # split relationship
    # print(sub_rel_points.size())
    # print(sub_rel_target.size())
    # print(sub_rel_idx.size())
    # print(sub_rel_mask.size())

    # for relationship split
    #print('point shape', sub_rel_points.shape)
    #print('target shape', sub_one_hot_rel_target.shape)
    #print('id shape', sub_rel_idx.shape, sub_rel_idx[:, :, 0])
    #print('mask shape', sub_rel_mask.shape, sub_rel_mask[:, :, 0])
    #print('relation mask shape', multi_batch_rel_mask.shape, multi_batch_rel_mask)

    sum_num = multi_batch_rel_mask.sum().cpu()
    # print('checkpoint multi_batch_rel_mask in tran_sg')

    N = len(multi_batch_rel_mask)
    rel_points = torch.zeros((sum_num, 1024, 4)).cuda()
    one_hot_rel_target = torch.zeros((sum_num, 27)).cuda()
    rel_idx = torch.zeros((sum_num, 2)).cuda()
    rel_mask = torch.zeros((sum_num, 1)).cuda()
    for i in range(N):
        start_idx = multi_batch_rel_mask[:i].sum().cpu()
        end_idx = multi_batch_rel_mask[:i + 1].sum().cpu()
        # print(start_idx, end_idx)
        rel_points[start_idx:end_idx, :, :] = sub_rel_points[i, :end_idx - start_idx, :, :]
        one_hot_rel_target[start_idx:end_idx, :] = sub_one_hot_rel_target[i, :end_idx - start_idx, :]
        rel_idx[start_idx:end_idx, :] = sub_rel_idx[i, :end_idx - start_idx, :]
        rel_mask[start_idx:end_idx, :] = sub_rel_mask[i, :end_idx - start_idx, :]
    input['one_hot_rel_target'] = rel_mask
    input['rel_mask'] = rel_mask
    input['rel_points'] = rel_points  # splited
    input['rel_idx'] = rel_idx
    return input


def split_obj(input):
    sub_object_points = input['object_point_set']
    sub_object_target = input['object_cls']
    sub_object_idx = input['object_idx']
    multi_batch_object_mask = input['multi_batch_object_mask']
    # for obj split
    # print('point shape', sub_object_points.shape)
    # print('target shape', sub_object_target.shape)
    # print('id shape', sub_object_idx.shape, sub_object_idx[:, :, 0])
    #print('object mask shape', multi_batch_object_mask.shape, multi_batch_object_mask)
    sum_num = multi_batch_object_mask.sum().cpu()
    object_points = torch.zeros((sum_num, 1024, 3)).cuda()
    object_target = torch.zeros((sum_num, 1)).cuda()
    object_idx = torch.zeros((sum_num, 1)).cuda()
    N = len(multi_batch_object_mask)
    for i in range(N):
        start_idx = multi_batch_object_mask[:i].sum().cpu()
        end_idx = multi_batch_object_mask[:i + 1].sum().cpu()
        # print('get: ',start_idx, end_idx) # for testing; test okay
        object_points[start_idx:end_idx, :, :] = sub_object_points[i, :end_idx - start_idx, :, :]
        object_target[start_idx:end_idx, :] = sub_object_target[i, :end_idx - start_idx, :]
        object_idx[start_idx:end_idx,:] = sub_object_idx[i,:end_idx - start_idx,:]
    # print('final object id', object_target.shape, object_points.shape, object_idx[:, 0])
    input['object_target'] = object_target
    input['object_points'] = object_points  # splited
    input['object_idx'] = object_idx
    return input

