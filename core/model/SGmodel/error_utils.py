import torch

def top_1_accurancy(output, target):
    batch_size = target.size(0)
    _, pred = torch.max(output, dim=1)
    pred = pred.t()
    # print(pred.type())
    # print(rel_target.type())
    correct = pred.eq(target.long().view(-1))
    # print(output.shape, target.shape, pred.shape, correct.shape, '  <<<  acc shape')

    accurancy = correct.float().mean()
    return accurancy


def top_n_accurancy(output, target, maxk=1):
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #print(output.shape, target.shape, pred.shape, pred.t().shape, '  <<<  topk acc shape')
    # print(pred.type())
    # print(rel_target.type())
    correct = pred.eq(target.long().view(1, -1).expand_as(pred))

    correct_k = correct[:maxk].view(-1).float().sum(0)
    rel_top_n_single = correct_k.mul_(1.0 / batch_size).data.cpu().item()  # not right? why mean
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
    ave_recall_n = recall_n.sum() / batch_size
    # print(ave_recall_n)
    return recall_n, ave_recall_n


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


# def calculate_kth_error(kths):  # TOO HARD TO BE DONE
#     filter_rel_pred = rel_pred.data.clone()
#     filter_rel_pred[:,0] = -10000
#     rel_pred_choice = filter_rel_pred.max(1)[1]
#     #print(rel_pred_choice.size())
#     #print(rel_target.size())
#     #print(softmax_rel_x.size())
#     select_idx = (rel_mask == 1).nonzero()[:,0]
#     #print(select_idx)
#     filter_rel_pred_choice = rel_pred_choice[select_idx]
#     filter_softmax_rel_x = softmax_rel_x[select_idx,:].clone()
#     filter_softmax_rel_x[:,0] = 0
#     filter_one_hot_rel_target = one_hot_rel_target[select_idx]

#     _, recall_1 = top_n_recall(filter_softmax_rel_x, filter_one_hot_rel_target, maxk=1)
#     _, recall_3 = top_n_recall(filter_softmax_rel_x, filter_one_hot_rel_target, maxk=3)
#     _, recall_5 = top_n_recall(filter_softmax_rel_x, filter_one_hot_rel_target, maxk=5)

#     #print(recall_1.data.cpu().numpy())
#     #print(recall_3.data.cpu().numpy())
#     #print(recall_5.data.cpu().numpy())


#     #rel_correct = filter_rel_pred_choice.eq(filter_rel_target.long().data).cpu().sum()
#     #rel_mean_correct.append(rel_correct.item() / float(filter_rel_target.size()[0]))
#     rel_top_1_acc.append(recall_1.data.cpu().numpy())
#     rel_top_3_acc.append(recall_3.data.cpu().numpy())
#     rel_top_5_acc.append(recall_5.data.cpu().numpy())
