import torch


# Transform Input to cuda
def transform_input(input, inside_name='input'):
    shape = ''
    if isinstance(input, torch.Tensor):
        shape = input.shape
        if isinstance(input, torch.DoubleTensor):
            input = input.float()  # tofloat
        if torch.cuda.is_available:
            input = input.cuda()  # tocuda
    elif isinstance(input, dict):
        for key in input.keys():
            input[key] = transform_input(input[key], inside_name + '.' + key)
        return input
    elif isinstance(input, list):
        for id, key in enumerate(input):
            input[id] = transform_input(input[id], inside_name + '[%d]' % id)
    elif isinstance(input, tuple):
        raise NotImplementedError('unable to change input %s' % inside_name)
    else:
        raise NotImplementedError(type(input))
    # print('transform_input - transform insidename = ', inside_name, shape)
    return input
