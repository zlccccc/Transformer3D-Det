import torch
import torch.nn.functional as F
from core.model.task_basemodel.base_model import base_module


class cls_module(base_module):
    def __init__(self):
        super(cls_module, self).__init__()

    def calculate_loss(self, input, output):
        gt = input['cls'].view(-1)
        out = output['value']
        output['cls_loss'] = F.cross_entropy(out, gt)

        loss = 0
        loss += output['cls_loss']
        output['loss'] = loss
        return output

    def calculate_error(self, input, output):
        gt = input['cls'].view(-1)
        out = output['value']
        maxpos = torch.argmax(out, dim=1)[0]  # TODO: change it
        output['instance_error'] = (maxpos != gt).sum().float()
        output['n_count'] = out.shape[0]

        output['error'] = output['instance_error']
        return output
