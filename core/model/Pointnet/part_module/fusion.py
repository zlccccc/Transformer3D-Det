import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ANS = \sum val * inputs
class OutputFusion(nn.Module):
    def __init__(self, n_input, initial=None, fusion_type='sum'):  # sum=1
        super(OutputFusion, self).__init__()
        self.n_input = n_input
        n_input -= 1
        self.fusion_type = fusion_type
        if initial is not None:
            assert n_input == len(initial), 'fusion n_input not right'
        else:
            initial = [1 / (n_input + 1) for i in range(n_input)]
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(initial), requires_grad=True)
        self.register_parameter('FuseionWeight',self.fuse_weight)

    def forward(self, inputs, is_cuda=True):
        # print(1-torch.sum(self.fuse_weight), self.fuse_weight)
        for i, val in enumerate(inputs):
            if i == 0:
                if self.fusion_type == 'sum':
                    ret = val * (1 - torch.sum(self.fuse_weight))
                else:
                    raise NotImplementedError('type %s not Implemented' % self.fusion_type)
            else:
                if self.fusion_type == 'sum':
                    ret += val * self.fuse_weight[i - 1]
                else:
                    raise NotImplementedError('type %s not Implemented' % self.fusion_type)
        return ret
