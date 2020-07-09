# __init__ codes
import torch.nn as nn
from abc import abstractmethod


class base_module(nn.Module):
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
        self.mode = 'train'
        self.train()

    def val_mode(self):
        self.mode = 'test'
        self.eval()

    def forward(self, input):
        output = self._forward(input)
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

    def set_params(self):
        lr_decay_mult = {}
        lr_decay_mult['nn.Conv2d'] = [1, 1, 2, 0]
        lr_decay_mult['nn.BatchNorm2d'] = [1, 0, 1, 0]
        arranged_names = set()
        for name, module in self.named_modules():
            module_trainable = False
            for key, value in lr_decay_mult.items():
                if isinstance(module, eval(key)):
                    if not module.weight.requires_grad:
                        continue
                    self.params.append({'params': module.weight, 'lr': value[0] * self.base_lr,
                                        'weight_decay': value[1] * self.weight_decay})
                    arranged_names.add(name + '.weight')
                    if module.bias is not None and len(value) == 4:
                        self.params.append({'params': module.bias, 'lr': value[2] * self.base_lr,
                                            'weight_decay': value[3] * self.weight_decay})
                        arranged_names.add(name + '.bias')
                module_trainable = True
            # print(name, type(module))
            if not module_trainable:
                print('params not set:', type(module), name)

        for name, param in self.named_parameters():
            if name not in arranged_names:
                self.params.append({'params': param})

        return self.params
