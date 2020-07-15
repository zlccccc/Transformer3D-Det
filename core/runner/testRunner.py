from core.utils.utils import save_checkpoint
from .runner_utils.testRunnerUtils import testmodel
import torch
import time
import traceback


def testRunner(info):
    config = info['config']
    optimizer = info['optimizer']
    model = info['model']
    loggers = info['loggers']
    lowest_error = info['lowest_error']
    last_iter = info['last_iter']
    t_start = time.time()
    if torch.cuda.is_available:
        print('use cuda(gpu)')
        model = model.cuda()
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    model.train_mode()  # change mode
    print('last_iter:', last_iter)
    model.val_mode()
    output_error = {}
    error, weight, test_time = [], [], 0.
    for testset_name, loader in info['testdataloaders'].items():
        _error, _weight = testmodel(model, loader, loggers, config.log_freq, testset_name)
        error.append(_error)
        weight.append(_weight)
        test_time += time.time() - t_start
        t_start = time.time()
        output_error[testset_name + '_error'] = _error
    error_final = sum(error) / sum(weight)  # calculate mean
    # for logger
    output_error['time'] = test_time
    output_error['test_time'] = test_time
    output_error['error'] = error_final
    output_error['prev_lowest_error'] = lowest_error
    output_error['flush'] = True
    output_error['n_count'] = 1
    loggers.update_error(output_error, True)  # similiar as model.val
    print('testing: done')
