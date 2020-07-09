import torch
import time


def testmodel(model, loader, loggers, test_freq, testset_name):
    # val_mode
    all_error, n_count = 0., 0
    for it, sample in enumerate(loader):
        if next(model.parameters()).is_cuda:
            for key in sample.keys():
                if isinstance(sample[key], torch.DoubleTensor):
                    sample[key] = sample[key].float()  # tofloat
                if isinstance(sample[key], torch.Tensor):
                    sample[key] = sample[key].cuda()
        output = model(sample)
        if it == 0:
            output['testset_name_out'] = testset_name
        output['iteration'] = [it + 1, len(loader)]
        if it == len(loader) - 1:  # TODO: 再改
            output['flush'] = True
        loggers.update_error(output, it % test_freq == 0 or it == len(loader) - 1)
        all_error += output['error']
        n_count += output['n_count']
    #print('testing one dataset : DONE', all_error / n_count)
    return all_error / n_count, 1.
