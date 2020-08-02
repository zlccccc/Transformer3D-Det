from core.utils.utils import save_checkpoint
from .runner_utils.testRunnerUtils import testmodel
import torch
import time
import traceback

# TODO
def iterRunner(info):
    config = info['config']
    train_loader_iter = iter(info['traindataloader'])
    optimizer = info['optimizer']
    lr_scheduler = info['lr_scheduler']
    model = info['model']
    loggers = info['loggers']
    lowest_error = info['lowest_error']
    last_iter = info['last_iter']
    t_start = time.time()
    T_START = time.time()
    if torch.cuda.is_available:
        print('use cuda(gpu)')
        model = model.cuda()
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    model.train_mode()  # change mode
    print('last_iter:', last_iter)
    for iter_id in range(last_iter + 1, config.max_iter + 1):
        for tries in range(100):
            try:
                input = next(train_loader_iter)
                break
            except Exception as e:
                if isinstance(e, StopIteration):
                    print('Start A New Epoch')
                else:
                    print('dataloader exception', str(e))
                    print(traceback.format_exc())
                train_loader_iter = iter(info['traindataloader'])
        if next(model.parameters()).is_cuda:
            for key in input.keys():
                if isinstance(input[key], torch.DoubleTensor):
                    input[key] = input[key].float()  # tofloat
                if isinstance(input[key], torch.Tensor):
                    input[key] = input[key].cuda()
        # TODO change zero_grad
        optimizer.zero_grad()
        t_loader = time.time()
        output = model(input)
        t_forward = time.time()
        assert 'loss' in output.keys(), 'Key "loss" should in output.keys'
        loss = output['loss']
        loss.backward()
        # model.average_gradients()  # multi card sync
        optimizer.step()
        # print('backward okay') # for test
        output['iteration'] = [iter_id, config.max_iter, (iter_id + 1) / len(info['traindataloader'])]
        output['loader_time'] = t_loader - t_start
        output['forward_time'] = t_forward - t_loader
        t_tmp = time.time()
        output['update_time'] = t_tmp - t_forward
        output['time'] = t_tmp - t_start
        output['mean_time_iter'] = (t_tmp - T_START) / (iter_id - last_iter)
        t_start = t_tmp
        if iter_id % config.test_freq == 0 or iter_id % config.save_freq == 0:
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
            is_best = error_final < lowest_error
            if is_best or iter_id % config.save_freq == 0:
                if is_best:
                    lowest_error = error_final
                save_checkpoint({
                    'step': iter_id,
                    'state_dict': model.state_dict(),
                    'lowest_error': lowest_error,
                    'optimizer': optimizer.state_dict(),
                }, is_best, config.snapshot_save_path + '/ckpt' + '_' + str(iter_id))
            model.train_mode()
        lr_scheduler.step()
        loggers.update_loss(output, iter_id % config.log_freq == 0)  # TODO
    print('training: done')
