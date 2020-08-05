import torch
import torch.nn.functional as F


def top_1_accurancy(output, target):
    # batch_size = target.size(0)
    _, pred = torch.max(output, dim=1)
    pred = pred.t()
    # print(pred.type(), rel_target.type())
    # print(pred, target.view(-1))
    correct = pred.eq(target.long().view(-1))
    # print(output.shape, target.shape, pred.shape, correct.shape, '  <<<  acc shape')
    accurancy = correct.float().mean()
    return accurancy


def top_n_accurancy(output, target, maxk=1):
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print(output.shape, target.shape, pred.shape, pred.t().shape, '  <<<  topk acc shape')
    # print(pred.type(), rel_target.type())
    correct = pred.eq(target.long().view(1, -1).expand_as(pred))
    correct_k = correct[:maxk].view(-1).float().sum(0)
    rel_top_n_single = correct_k.mul_(1.0 / batch_size).data.cpu().item()  # not right? why mean
    return rel_top_n_single


def top_n_recall(output, target, maxk):  # recall ???
    # print('calculating recall', output.shape, target.shape, maxk)
    # output = F.softmax(output)  # not necessary
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred_onehot = F.one_hot(pred, num_classes=output.shape[-1])
    pred_onehot = pred_onehot.sum(dim=0).float()
    recall_n = (pred_onehot * target).sum(dim=-1).float() / target.sum(dim=-1).float()
    ave_recall_n = recall_n.sum() / batch_size
    # print(ave_recall_n)
    # print(recall_n.shape, ave_recall_n)
    return ave_recall_n


def calculate_recall_per_edge(object_idx, object_predict, object_target, rel_idx, rel_mask, rel_predict, one_hot_rel_target, maxk):
    # one_hot_rel_target, rel_idx, rel_mask, object_idx, object_target, softmax_object_x, softmax_rel_x,  prediction_recall=50
    # print(rel_mask)
    # print(select_idx)
    # print(object_target.size())
    # print(object_idx.size())
    # print(softmax_object_x.size())
    object_predict = F.softmax(object_predict)
    rel_predict = F.sigmoid(rel_predict)  # should? todo checkit

    recall_per_edge = []
    for i in range(len(rel_idx)):
        #  for one edge
        rel_pred = rel_predict[i]  # in this option
        rel_gt = (one_hot_rel_target == 1)
        # get sub and obj id
        sub_id = rel_idx[i][0]
        obj_id = rel_idx[i][1]

        sub_index = torch.where(object_idx == sub_id)
        sub_gt = object_target[sub_index]
        sub_pred = object_predict[sub_index]

        obj_index = torch.where(object_idx == obj_id)
        obj_gt = object_target[obj_index]
        obj_pred = object_predict[obj_index]

        print(sub_gt.shape, sub_pred.shape, obj_gt.shape, obj_pred.shape, rel_pred.shape, rel_gt.shape, '   <<<<<<<<<<< sub and obj shape')
        # for object:
        #   correct = pred.eq(target.long().view(1, -1).expand_as(pred))
        #   correct_k = correct[:maxk].view(-1).float().sum(0)
        #   rel_top_n_single = correct_k.mul_(1.0 / batch_size).data.cpu().item()  # not right? why mean
        # for relation:
        #   pred_onehot = F.one_hot(pred, num_classes=output.shape[-1])
        #   pred_onehot = pred_onehot.sum(dim=0).float()
        #   recall_n = (pred_onehot * target).sum(dim=-1).float() / target.sum(dim=-1).float()
        #   ave_recall_n = recall_n.sum() / batch_size
        pred_relation = torch.mm(obj_pred.unsqueeeze(1), sub_pred.unsqueeze(0))
        pred_relation = torch.mm(pred_relation.unsqueeeze(2), rel_pred.unsqueeze(0))
        print(pred_relation.shape)
        pred_relation = pred_relation.view(-1)
        _, pred_id = pred_relation.topk(maxk, 1, True, True)
        # TODO : reshape and calculate error
        recall_per_edge.append(0)

    # print(recall_50_per_graph), print(recall_100_per_graph)
    # print('------------------------')
    exit()
    return sum(recall_per_edge) / len(recall_per_edge)


def calculate_kth_error(predict, target, top_k, error_type='accurancy'):  # mean loss
    if error_type == 'accurancy':
        return top_n_accurancy(predict, target, top_k)
    elif error_type == 'recall':
        return top_n_recall(predict, target, top_k)
    elif error_type == 'top_accurancy':
        assert top_k == 1
        return top_1_accurancy(predict, target)
    else:
        raise NotImplementedError(error_type)
