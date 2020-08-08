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


def calculate_recall_per_edge(object_predict, object_target, rel_predict, one_hot_rel_target, input, maxk):
    # one_hot_rel_target, rel_idx, rel_mask, object_idx, object_target, softmax_object_x, softmax_rel_x,  prediction_recall=50
    # print(rel_mask)
    # print(select_idx)
    # print(object_target.size())
    # print(object_idx.size())
    # print(softmax_object_x.size())
    # REMOVE ZERO EDGE
    #rel_predict = rel_predict[:, 1:]
    #one_hot_rel_target = one_hot_rel_target[:, 1:]
    object_predict = F.softmax(object_predict)
    rel_predict = F.sigmoid(rel_predict)  # should? todo checkit
    #print(object_predict.shape, rel_predict.shape, rel_idx.shape)
    multi_batch_object_mask = input['multi_batch_object_mask']
    multi_batch_rel_mask = input['multi_batch_rel_mask']
    sub_object_idx, sub_rel_idx, rel_mask = input['object_idx'], input['rel_idx'], input['rel_mask']

    #print(sub_object_idx.shape, rel_mask.shape)
    # sub_id = sub_rel_idx[:, 0]
    # obj_id = sub_rel_idx[:, 1]

    recall_per_edge = []
    N = len(multi_batch_rel_mask)
    # print('N:', N)
    for i in range(N):
        object_idx_scene = sub_object_idx[i]
        rel_idx_scene = sub_rel_idx[i]

        start_idx = multi_batch_rel_mask[:i].sum().cpu()
        end_idx = multi_batch_rel_mask[:i + 1].sum().cpu()
        rel_pred_scene = rel_predict[start_idx:end_idx, :]
        rel_tar_scene = one_hot_rel_target[start_idx:end_idx, :]
        rel_mask_scene = rel_mask[start_idx:end_idx, :]
        #print('start and end', start_idx, end_idx, multi_batch_rel_mask[i], rel_predict.shape, len(rel_pred_scene))

        start_idx = multi_batch_object_mask[:i].sum().cpu()
        end_idx = multi_batch_object_mask[:i + 1].sum().cpu()
        obj_pred_scene = object_predict[start_idx:end_idx, :]
        obj_tar_scene = object_target[start_idx:end_idx, :]
        # print(sub_object_idx.shape, sub_rel_idx.shape, 'objshape', obj_pred.shape,'  <<<  view',i)  # TODO CHANGE IT
        # recall_per_edge = []
        # for i in range(N):
        #     start_idx = multi_batch_object_mask[:i].sum().cpu()
        for id in range(multi_batch_rel_mask[i]):
            if rel_mask_scene[id] == 0:
                #assert rel_tar_scene[id].sum() == 0, 'relation mask count not right'
                # print(rel_tar_scene[id].sum())
                continue
            else:
                #assert rel_tar_scene[id].sum() != 0, 'relation count not right'
                pass
            #  for one edge
            rel_pred = rel_pred_scene[id]  # in this option
            rel_gt = rel_tar_scene[id]
            # get sub and obj id
            sub_id = rel_idx_scene[id][0]
            obj_id = rel_idx_scene[id][1]
            #print(object_idx, sub_id, obj_id, (object_idx == obj_id).nonzero(),'obj ID <<<< ')
            #print(torch.where(object_idx == sub_id))

            sub_index = (object_idx_scene == sub_id).nonzero()
            assert len(sub_index)==1, 'sub_index should be one-one'
            #print(sub_index[0,0])
            sub_gt = obj_tar_scene[sub_index[0,0], :]
            #print(sub_gt, obj_tar[1],'<< subshape')
            sub_pred = obj_pred_scene[sub_index[0,0], :]

            obj_index = (object_idx_scene == obj_id).nonzero()
            assert len(obj_index)==1, 'obj_index should be one-one'
            obj_gt = obj_tar_scene[obj_index[0,0], :]
            obj_pred = obj_pred_scene[obj_index[0,0], :]
            #print(object_idx, obj_index, sub_index, sub_id, obj_id, '   <<<    subindex')

            #print(sub_gt.shape, sub_pred.shape, obj_gt.shape, obj_pred.shape, 'rel', rel_pred.shape, rel_gt.shape, '   <<<<<<<< sub and obj shape')
            #print(sub_pred.shape, obj_pred.shape, 'rel', rel_pred.shape, rel_gt.shape, '   <<<pred<<< sub and obj shape')
            #print(sub_pred.shape, obj_pred.unsqueeze(1).shape, 'rel', rel_pred.shape, rel_gt.shape, '   <<<pred<<< sub and obj shape')
            # for object:
            #   correct = pred.eq(target.long().view(1, -1).expand_as(pred))
            #   correct_k = correct[:maxk].view(-1).float().sum(0)
            #   rel_top_n_single = correct_k.mul_(1.0 / batch_size).data.cpu().item()  # not right? why mean
            # for relation:
            #   pred_onehot = F.one_hot(pred, num_classes=output.shape[-1])
            #   pred_onehot = pred_onehot.sum(dim=0).float()
            #   recall_n = (pred_onehot * target).sum(dim=-1).float() / target.sum(dim=-1).float()
            #   ave_recall_n = recall_n.sum() / batch_size
            pred_relation = torch.mm(rel_pred.view(-1,1), obj_pred.view(1,-1))
            #print(pred_relation.shape, '<< predrelationshape 1')
            pred_relation = torch.mm(pred_relation.view(-1,1), sub_pred.view(1,-1))
            #print(pred_relation.shape, '<< relationship 2')
            pred_relation = pred_relation.view(-1)
            _, pred_id = pred_relation.topk(maxk, 0, True, True)
            l_obj = len(obj_pred)
            # TODO : reshape and calculate error
            right, all_rel = 0, float(rel_gt.sum())
            for i in range(maxk):
                now_id_ = pred_id[i]
                rel_id_ = now_id_ / l_obj / l_obj
                obj_id_ = now_id_ / l_obj % l_obj
                sub_id_ = now_id_ % l_obj
                if obj_gt == obj_id_ and sub_gt == sub_id_:
                    #print(obj_gt, sub_gt, '<< obj and sub ok')
                    right += rel_gt[rel_id_]
            #print(right / all_rel, '  <<  percent', maxk)
            recall_per_edge.append(right / all_rel)

    # print(recall_50_per_graph), print(recall_100_per_graph)
    # print('------------------------')
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
