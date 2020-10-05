from core.utils.utils import save_checkpoint
from .runner_utils.utils import transform_input
from .runner_utils.testRunnerUtils import testmodel
import torch
import torch.nn as nn
import time
import traceback


def print_grad(model, keyword=None):
    print('Calculate grad')
    for name, param in model.named_parameters():
        # if 'weight' not in name:
        #     continue
        if keyword is not None and keyword not in name:
            continue
        print(name, 'max: grad[%.5f] value[%.5f]' %(float(torch.max(param.grad).cpu()), float(torch.max(param.data).cpu())), end=' ; ')
        print('std: grad[%.5f] value[%.5f]' %(float(param.grad.detach().std().cpu()), float(param.data.detach().std().cpu())), 'shape', param.shape, flush=True)
        # print('real value', param.grad.cpu()[:3, :], param.data.cpu()[:3, :], flush=True)


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


def iterBNDecayRunner(info):
    config = info['config']
    train_loader_iter = iter(info['traindataloader'])
    optimizer = info['optimizer']
    lr_scheduler = info['lr_scheduler']
    model = info['model']
    loggers = info['loggers']
    lowest_error = info['lowest_error']
    last_iter = info['last_iter']
    # BN_MOMEMTUM
    # BN_DECAY_STEP = FLAGS.bn_decay_step
    # BN_DECAY_RATE = FLAGS.bn_decay_rate
    BN_DECAY_STEP = 3000  #20*150
    BN_DECAY_RATE = 0.5
    BN_MOMENTUM_INIT = 0.5
    BN_MOMENTUM_MAX = 0.001
    bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
    bnm_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=last_iter)
    t_start = time.time()
    T_START = time.time()
    if isinstance(model, torch.nn.DataParallel):
        model.module.train_mode()
    elif isinstance(model, torch.nn.Module):
        model.train_mode()  # change mode
    else:
        raise NotImplementedError(type(model))
    print('last_iter:', last_iter)
    max_tries = 3
    for iter_id in range(last_iter + 1, config.max_iter + 1):
        lr_scheduler.step()
        bnm_scheduler.step()
        for tries in range(max_tries):
            try:
                input = next(train_loader_iter)
                break
            except Exception as e:
                if isinstance(e, StopIteration):
                    print('Start A New Epoch', flush=True)
                    print('Current BN decay momentum: %f' % (bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
                else:
                    if tries == max_tries - 1:
                        raise e
                    print('dataloader exception', str(e))
                    print(traceback.format_exc())
                train_loader_iter = iter(info['traindataloader'])
        input = transform_input(input)
        optimizer.zero_grad()
        t_loader = time.time()
        output = model(input)  # also could backward inside
        t_forward = time.time()
        if isinstance(model, torch.nn.DataParallel):
            # mutli-batch; for data-parallel-model use
            for key, value in output.items():
                if 'loss' in key:
                    output[key] = torch.mean(value, dim=0)
        assert 'loss' in output.keys(), 'Key "loss" should in output.keys'
        loss = output['loss']
        # print(loss)
        loss.backward()
        # model.average_gradients()  # multi card sync
        # print_grad(model, 'weight')
        # if iter_id % 1000 == 0:# or True:# and False:  # just print
        # if True:# and False:  # just print
        #     print_grad(model, '.0.weight')  # conv_first
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
        output['lr'] = lr_scheduler.get_lr()[0]
        if iter_id % config.test_freq == 0 or iter_id % config.save_freq == 0:
            if isinstance(model, torch.nn.DataParallel):
                model.module.val_mode()
            elif isinstance(model, torch.nn.Module):
                model.val_mode()  # change mode
            else:
                raise NotImplementedError(type(model))
            output_error = {}
            error, weight, test_time = [], [], 0.
            for testset_name, loader in info['testdataloaders'].items():
                _error, _weight = testmodel(model, loader, loggers, config.log_freq, testset_name, iter_id)
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
            if isinstance(model, torch.nn.DataParallel):
                model.module.train_mode()
            elif isinstance(model, torch.nn.Module):
                model.train_mode()  # change mode
            else:
                raise NotImplementedError(type(model))
        loggers.update_loss(output, iter_id % config.log_freq == 0)  # TODO
    print('training: done')
