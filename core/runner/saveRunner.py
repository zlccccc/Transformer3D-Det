import torch
import time


def saveRunner(info):
    model = info['model']
    loggers = info['loggers']
    t_start = time.time()
    if torch.cuda.is_available:
        print('use cuda(gpu)')
        model = model.cuda()
    # change mode
    if isinstance(model, torch.nn.DataParallel):
        model.module.test_mode()
    elif isinstance(model, torch.nn.Module):
        model.test_mode()  # change mode
    else:
        raise NotImplementedError(type(model))

    assert hasattr(model, 'save_dataset'), 'you should save dataset in model'
    for testset_name, loader in info['testdataloaders'].items():
        if 'save' not in testset_name:
            continue
        model.save_dataset(loader, loggers)
        print('Testing Dataset: use %.5fs' % (time.time() - t_start))
        t_start = time.time()
    # for logger
    print('testing: done')
