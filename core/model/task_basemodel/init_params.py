import math
import torch.nn as nn
from torch.nn import init


def init_params(m, BatchNorm2d, init_type, nonlinearity):
    init_type = init_type.split('.')
    assert len(init_type) <= 2, 'init_type.length <= 2'
    init_cnn, init_linear = init_type[0], init_type[-1]
    a = 0.25
    if isinstance(m, nn.Conv2d):
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(m.weight.data)
        gain = math.sqrt((fan_in+fan_out/2/fan_in))
        if init_cnn == 'kaiming_uniform':
            init.kaiming_uniform_(m.weight.data, a, nonlinearity=nonlinearity)
        elif init_cnn == 'kaiming_normal':
            init.kaiming_normal_(m.weight.data, a, nonlinearity=nonlinearity)
        elif init_cnn == 'kaiming_uniform_fan_out':
            init.kaiming_uniform_(m.weight.data, a, mode='fan_out', nonlinearity=nonlinearity)
        elif init_cnn == 'kaiming_normal_fan_out':
            init.kaiming_normal_(m.weight.data, a, mode='fan_out', nonlinearity=nonlinearity)
        elif init_cnn == 'xavier_uniform':
            init.xavier_uniform_(m.weight.data, gain)
        elif init_cnn == 'xavier_normal':
            init.xavier_normal_(m.weight.data, gain)
        elif init_cnn == 'init':
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        else:
            raise NotImplementedError(init_cnn)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, BatchNorm2d):
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(m.weight.data)
        gain = math.sqrt((fan_in+fan_out)/2/fan_in)
        if init_linear == 'kaiming_uniform':
            init.kaiming_uniform_(m.weight.data, a, nonlinearity=nonlinearity)
        elif init_linear == 'kaiming_normal':
            init.kaiming_normal_(m.weight.data, a, nonlinearity=nonlinearity)
        elif init_linear == 'kaiming_uniform_fan_out':
            init.kaiming_uniform_(m.weight.data, a, mode='fan_out', nonlinearity=nonlinearity)
        elif init_linear == 'kaiming_normal_fan_out':
            init.kaiming_normal_(m.weight.data, a, mode='fan_out', nonlinearity=nonlinearity)
        elif init_linear == 'xavier_uniform':
            init.xavier_uniform_(m.weight.data, gain)
        elif init_linear == 'xavier_normal':
            init.xavier_normal_(m.weight.data, gain)
        elif init_linear == 'init':
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
        else:
            raise NotImplementedError(init_linear)
        init.constant_(m.bias.data, 0)


# weight_init(same as top few lines)
def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)