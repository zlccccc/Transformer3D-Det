import torch
import time


def testmodel(model, loader, loggers, test_freq, testset_name, last_iter):  # last_iter: for logging use
    # must be val_mode
    all_error, n_count = 0., 0
    for it, sample in enumerate(loader):
        if next(model.parameters()).is_cuda:
            for key in sample.keys():
                if isinstance(sample[key], torch.DoubleTensor):
                    sample[key] = sample[key].float()
                if isinstance(sample[key], torch.Tensor):
                    sample[key] = sample[key].cuda()
        output = model(sample)
        # mutli-batch; for data-parallel-model use
        for key, value in output.items():
            if 'error' in key or 'n_count' == key:
                output[key] = torch.sum(value, dim=0)
            # print('error', key, value.shape, output[key].shape)
        # print('error sum', output['error'])
        if it == 0:
            output['testset_name_out'] = testset_name
        output['iteration'] = [it + 1, len(loader), (it + 1) / len(loader)]
        if it == len(loader) - 1:
            output['last_iter'] = last_iter
            output['flush'] = True
        loggers.update_error(output, it % test_freq == 0 or it == len(loader) - 1)
        all_error += output['error']
        n_count += output['n_count']
    # print('testing one dataset : DONE', all_error / n_count)
    return all_error / n_count, 1.
