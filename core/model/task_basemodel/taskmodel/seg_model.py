import torch
import torch.nn.functional as F
from core.model.task_basemodel.backbone.base_model import base_module


class seg_module(base_module):
    def __init__(self):
        super(seg_module, self).__init__()

    def calculate_loss(self, input, output):
        gt = input['seg']
        out = output['value']
        gt = gt.view(gt.shape[0], gt.shape[1])
        # print(out.shape, gt.shape, '<<< shape')
        # out = torch.log_softmax(out, dim=-1)
        # output['seg_loss'] = F.nll_loss(out, gt)
        # TODO: CHECK IT (change error calculating)
        _out, _gt = out.contiguous().view(-1, out.shape[-1]), gt.view(-1)
        # print(_out.shape, _gt.shape, '<<< out shape')
        output['seg_loss'] = F.cross_entropy(_out, _gt)
        n_count = out.shape[0] * out.shape[1]
        maxpos = torch.argmax(out, dim=-1)
        maxpos = out.data.max(-1)[1]
        # print(maxpos.shape, gt.shape, 'pos shape')
        output['accurancy(loss)'] = (maxpos == gt).sum().float() / n_count

        loss = 0
        loss += output['seg_loss']
        output['loss'] = loss
        return output

    def calculate_error(self, input, output):
        gt = input['seg']
        out = output['value']
        gt = gt.view(gt.shape[0], gt.shape[1])

        maxpos = torch.argmax(out, dim=-1)
        maxpos = out.data.max(-1)[1]
        # print(maxpos.shape, gt.shape, 'pos shape')
        n_count = out.shape[0] * out.shape[1]
        output['accurancy(error)'] = (maxpos == gt).sum().float() / n_count
        output['accurancy(error_count)'] = 1
        # maxpos = out.data.max(1)[1]  #same
        # print(out.data.max(1)[1], torch.argmax(out, dim=1))

        output['error'] = 1 - output['accurancy(error)']
        output['n_count'] = 1
        return output
