import argparse
import os
import sys

sys.path.append("./")  # for debuggingfrom core.runner import getrunner
from core.data import get_dataset, get_one_dataset
from core.model import model_entry
from core.other.optimizer import get_optimizer
from core.other.lr_scheduler import get_lr_scheduler
from core.other.logs import Loggers
from core.utils.utils import load_state
from core.runner import getrunner

import yaml
from easydict import EasyDict
import torch
import torch.utils.data as data
# print(os.getcwd())

parser = argparse.ArgumentParser(description='PyTorch training script')
parser.add_argument('--config', default='None', type=str, help='config yaml path')
parser.add_argument('--test', action='store_true', default=False, help='use testset to test model')
parser.add_argument('--opt', default='None', type=str, help='other option')
parser.add_argument('--gpu', default='0', type=str, help='gpu device')


def main():
    args = parser.parse_args()
    torch.backends.cudnn.enabled = False
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # print('config', args.config)
    with open(args.config, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    if args.test:
        config.train.runner.name = 'test'
        config.common.load.load = True
    loggers = Loggers(config.common.logs)
    loggers.update_loss({'args_out': args, 'config_out': config}, True)

    train_dataset = get_one_dataset(config.train.dataset)
    test_datasets = get_dataset(config.test.dataset)

    model = model_entry(config.common.model)
    # TO CHANGE BASE_LR AND WEIGHT_DECAY (group parameters)
    base_lr = config.train.lr_scheduler.base_lr
    weight_decay = config.train.optimizer.weight_decay
    parameters = model.set_params(base_lr, weight_decay)  # for grouping
    config.train.optimizer['lr'] = base_lr  # for base use (not grouped)
    optimizer = get_optimizer(config.train.optimizer, parameters)
    # load model pa rams
    lowest_err, last_iter = float('inf'), -1
    if config.common.load.load:  # load model
        load_path = config.common.load.path
        print('load model from %s' % load_path)
        if not os.path.exists(load_path):
            raise AssertionError('load_path not exist')
        load_way = config.common.load.get('type', 'recover')
        if load_way == 'recover':
            print('Resume training from a previous checkpoint ...')
            lowest_err, last_iter = load_state(load_path, model, optimizer=optimizer)
        elif load_way == 'finetune':
            print('Finetuning from a previous model ...')
            load_state(load_path, model)
        else:
            raise NotImplementedError('load_way: %s' % load_way)
    else:
        print('Start new training')
    config.train.lr_scheduler['optimizer'] = optimizer
    config.train.lr_scheduler['last_iter'] = last_iter
    lr_scheduler = get_lr_scheduler(config.train.lr_scheduler)
    traindataloader = data.DataLoader(train_dataset, batch_size=config.train.batch_size,
                                      shuffle=True, num_workers=config.train.workers, drop_last=True,
                                      pin_memory=True)
    testdataloaders = {}
    for key, value in test_datasets.items():
        testdataloaders[key] = data.DataLoader(value, batch_size=config.test.batch_size,
                                               shuffle=False, num_workers=config.test.workers, drop_last=False,
                                               pin_memory=True)

    # TODO: MULTI GPU TODO!!!
    # if torch.cuda.device_count() > 1:
    #     if (args.syncBN == True):
    #         torch.cuda.set_device(args.local_rank)
    #         world_size = args.ngpu
    #         torch.distributed.init_process_group(
    #             'nccl',
    #             init_method='env://',
    #             world_size=world_size,
    #             rank=args.local_rank,
    #         )
    #         #classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    #         device = torch.device('cuda:{}'.format(args.local_rank))
    #         model.to(device)
    #         model = torch.nn.parallel.DistributedDataParallel(
    #             model,
    #             device_ids=[args.local_rank],
    #             output_device=args.local_rank,
    #         )
    #     else:
    #         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #         print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         model = torch.nn.DataParallel(model)
    #         model.to(device)

    if torch.cuda.is_available:
        print('use cuda(gpu)')
        model = model.cuda()
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    info = {
        'config': config.train.runner,
        'traindataloader': traindataloader,
        'testdataloaders': testdataloaders,
        'lr_scheduler': lr_scheduler,
        'optimizer': optimizer,
        'lowest_error': lowest_err,
        'loggers': loggers,
        'model': model,
        'last_iter': last_iter,
    }
    runner = getrunner(config.train.runner)
    runner(info)


if __name__ == "__main__":
    main()
