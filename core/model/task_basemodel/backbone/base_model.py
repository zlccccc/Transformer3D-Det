# __init__ codes
import torch
import torch.nn as nn
from abc import abstractmethod


class base_module(nn.Module):
    '''backbone
    _before_forward -> _forward -> _after_forward
    return calculate_loss / calculate_error / origin
    '''

    def __init__(self):
        self.mode = None
        super(base_module, self).__init__()

    @abstractmethod
    def calculate_loss(self, input, output):
        pass

    @abstractmethod
    def calculate_error(self, input, output):
        pass

    @abstractmethod
    def _forward(self, input):
        pass

    def train_mode(self):
        if torch.cuda.is_available: # not useful?
            torch.cuda.empty_cache()
        self.mode = 'train'
        self.train()

    def val_mode(self):
        if torch.cuda.is_available: # not useful?
            torch.cuda.empty_cache()
        self.mode = 'test'
        self.eval()

    @abstractmethod
    def _before_forward(self, input):
        return input

    @abstractmethod
    def _after_forward(self, input):
        return input

    def forward(self, input):
        input = self._before_forward(input)
        # _before_forward = getattr(self, '_before_forward')  # same
        # print('before_forward: done')
        output = self._forward(input)
        input = self._after_forward(input)
        if self.mode == 'train':
            return self.calculate_loss(input, output)
        elif self.mode == 'test':
            return self.calculate_error(input, output)
        elif self.mode == 'val':
            return output
        else:
            raise NotImplementedError(self.mode)

    def init_params(self, BatchNorm2d, init_type):
        from core.model.task_basemodel.init_params import init_params
        # from init_params import init_params
        print('init model params using %s' % init_type)
        for m in self.modules():
            init_params(m, BatchNorm2d, init_type, nonlinearity=self.init_relu)

    def set_params_lr(self,base_lr, weight_decay):
        parameters = []
        lr_decay_mult = {}
        lr_decay_mult['nn.Conv2d'] = [1, 1, 2, 0]  # weight and bias mult
        lr_decay_mult['nn.Conv1d'] = [1, 1, 2, 0]  # weight and bias mult
        lr_decay_mult['nn.BatchNorm2d'] = [1, 0, 1, 0]
        lr_decay_mult['nn.BatchNorm1d'] = [1, 0, 1, 0]
        print('set param lr using lr_decay_mult')
        print(lr_decay_mult)
        arranged_names = set()
        for name, module in self.named_modules():
            module_trainable = False
            for key, value in lr_decay_mult.items():
                if isinstance(module, eval(key)):
                    if not module.weight.requires_grad:
                        continue
                    parameters.append({'params': module.weight, 'lr': value[0] * base_lr,
                                       'weight_decay': value[1] * weight_decay})
                    arranged_names.add(name + '.weight')
                    print('set parameter', 'lr:',value[0], 'weight_decay:',value[1], name + '.weight', key)
                    if module.bias is not None and len(value) == 4:
                        parameters.append({'params': module.bias, 'lr': value[2] * base_lr,
                                           'weight_decay': value[3] * weight_decay})
                        arranged_names.add(name + '.bias')
                        print('set parameter', 'lr:',value[2], 'weight_decay:',value[3], name + '.bias', key)
                module_trainable = True
            # print(name, type(module))
            if not module_trainable:
                print('params not set(not trainable):', type(module), name)

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name not in arranged_names:
                print('set parameter (default)', name)
                self.params.append({'params': param})  # 默认parameter group(base lr; parameter group)
        print('get params end')
        return parameters

    def set_params(self, base_lr, weight_decay, weight_type='base'):
        if weight_type == 'group':
            return self.set_params_lr(base_lr, weight_decay)
        elif weight_type == 'base':
            return self.parameters()
        else:
            raise NotImplementedError(weight_type)
