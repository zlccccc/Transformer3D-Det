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
    sum_num = sum(multi_batch_rel_mask.data.numpy())
    multi_batch_rel_mask = multi_batch_rel_mask.data.numpy()
    multi_batch_rel_mask = np.insert(multi_batch_rel_mask, 0, 0)
    # print(sum_num)
    # print('checkpoint multi_batch_rel_mask in tran_sg')
    # print(multi_batch_rel_mask)
    N = len(multi_batch_rel_mask)
    rel_points = np.zeros((sum_num, 1024, 4))
    one_hot_rel_target = np.zeros((sum_num, 27))
    rel_idx = np.zeros((sum_num, 2))
    rel_mask = np.zeros((sum_num, 1))

    for i in range(0, N - 1):
        start_idx = sum(multi_batch_rel_mask[:i + 1])
        end_idx = sum(multi_batch_rel_mask[:i + 2])
        # print(start_idx, end_idx)
        rel_points[start_idx:end_idx, :, :] = sub_rel_points.data.numpy()[i, :end_idx - start_idx, :, :]
        one_hot_rel_target[start_idx:end_idx, :] = sub_one_hot_rel_target.data.numpy()[i, :end_idx - start_idx, :]
        rel_idx[start_idx:end_idx, :] = sub_rel_idx.data.numpy()[i, :end_idx - start_idx, :]
        rel_mask[start_idx:end_idx, :] = sub_rel_mask.data.numpy()[i, :end_idx - start_idx, :]
    one_hot_rel_target = torch.Tensor(one_hot_rel_target)
    rel_mask = torch.Tensor(rel_mask)
    rel_points = torch.Tensor(rel_points)
    rel_idx = torch.Tensor(rel_idx)
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

    sum_num = sum(multi_batch_object_mask.data.numpy())
    # print('checkpoint multi_batch_object_mask in train_sg')
    # print(multi_batch_object_mask)
    object_points = np.zeros((sum_num, 1024, 3))
    object_target = np.zeros((sum_num, 1))
    object_idx = np.zeros((sum_num, 1))
    multi_batch_object_mask = multi_batch_object_mask.data.numpy()
    multi_batch_object_mask = np.insert(multi_batch_object_mask, 0, 0)
    # print(multi_batch_object_mask)
    N = len(multi_batch_object_mask)
    # print(sub_object_points.shape)
    # print(sub_object_target.shape)
    # print(sub_object_idx.shape)
    for i in range(0, N - 1):
        start_idx = sum(multi_batch_object_mask[:i + 1])
        end_idx = sum(multi_batch_object_mask[:i + 2])
        # print(start_idx, end_idx)
        object_points[start_idx:end_idx, :, :] = sub_object_points.data.numpy()[i, :end_idx - start_idx, :, :]
        object_target[start_idx:end_idx, :] = sub_object_target.data.numpy()[i, :end_idx - start_idx, :]
        object_idx[start_idx:end_idx, :] = sub_object_idx.data.numpy()[i, :end_idx - start_idx, :]
    object_target = torch.Tensor(object_target)
    object_points = torch.Tensor(object_points)
    object_idx = torch.Tensor(object_idx)
    input['object_target'] = object_target
    input['object_points'] = object_points  # splited
    input['object_idx'] = object_idx
    return input


def calculate_loss(predict, target, loss_type):
    if loss_type == 'l1':
        object_loss = F.nll_loss(predict, target.long(), object_trans_feat)
    elif loss_type == 'focal_loss':
        one_hot_target = F.one_hot(target.long(), num_classes=object_num_class)
        object_loss = py_sigmoid_focal_loss(predict, one_hot_target)
    else:
        raise NotImplementedError(loss_type)
    return object_loss


def top_n_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print(pred.type())
    # print(rel_target.type())
    correct = pred.eq(target.long().view(1, -1).expand_as(pred))
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        rel_top_n_single = correct_k.mul_(1.0 / batch_size).data.cpu().item()
    return rel_top_n_single


def top_n_recall(output, target, maxk):  # recall ???
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    # print(output[-5])
    # print(_[-5])
    # print(pred[-5])
    pred = pred.t()
    #print('pred with t')
    # print(pred[:,0])
    pred_onehot = F.one_hot(pred, num_classes=27)
    # print(pred_onehot.size())
    pred_onehot = pred_onehot.sum(dim=0).float()

    # print('pred')
    # print(pred_onehot[-5])
    # print('target')
    # print(target[-5])
    # print(pred_onehot.size())
    # print(target.size())
    recall_n = (pred_onehot * target).sum(dim=-1).float()/target.sum(dim=-1).float()
    # print('recall_n')
    # print(recall_n.size())
    ave_recall_n = recall_n.sum()/batch_size
    # print(ave_recall_n)
    return recall_n, ave_recall_n


def calculate_kth_error(kths): # TO BE DONE
    filter_rel_pred = rel_pred.data.clone()
    filter_rel_pred[:,0] = -10000
    rel_pred_choice = filter_rel_pred.max(1)[1]
    #print(rel_pred_choice.size())
    #print(rel_target.size())
    #print(softmax_rel_x.size())
    select_idx = (rel_mask == 1).nonzero()[:,0]
    #print(select_idx)
    filter_rel_pred_choice = rel_pred_choice[select_idx]
    filter_softmax_rel_x = softmax_rel_x[select_idx,:].clone()
    filter_softmax_rel_x[:,0] = 0
    filter_one_hot_rel_target = one_hot_rel_target[select_idx]

    _, recall_1 = top_n_recall(filter_softmax_rel_x, filter_one_hot_rel_target, maxk=1)
    _, recall_3 = top_n_recall(filter_softmax_rel_x, filter_one_hot_rel_target, maxk=3)
    _, recall_5 = top_n_recall(filter_softmax_rel_x, filter_one_hot_rel_target, maxk=5)

    #print(recall_1.data.cpu().numpy())
    #print(recall_3.data.cpu().numpy())
    #print(recall_5.data.cpu().numpy())


    #rel_correct = filter_rel_pred_choice.eq(filter_rel_target.long().data).cpu().sum()
    #rel_mean_correct.append(rel_correct.item() / float(filter_rel_target.size()[0]))
    rel_top_1_acc.append(recall_1.data.cpu().numpy())
    rel_top_3_acc.append(recall_3.data.cpu().numpy())
    rel_top_5_acc.append(recall_5.data.cpu().numpy())


    
    #print(filter_rel_pred_choice.size())
    #print(filter_rel_target.size())
    #print(filter_softmax_rel_x.size())


def calculate_loss(input, kth, output):

    pass
