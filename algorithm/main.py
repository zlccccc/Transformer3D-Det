import argparse
import os
import sys

import yaml
from easydict import EasyDict
import torch
import torch.utils.data as data
sys.path.append("./")  # for debuggingfrom core.runner import getrunner
# print(os.getcwd())
from core.runner import getrunner
from core.utils.utils import load_state
from core.other.logs import Loggers
from core.other.lr_scheduler import get_lr_scheduler
from core.other.optimizer import get_optimizer
from core.model import model_entry
from core.data import get_dataset, get_one_dataset

parser = argparse.ArgumentParser(description='PyTorch training script')
parser.add_argument('--config', default='None', type=str)
parser.add_argument('--opt', default='None', type=str)
parser.add_argument('--gpu', default='0', type=str)


def main():
    args = parser.parse_args()
    torch.backends.cudnn.enabled = False
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # TODO: GPU
    # print('config', args.config)
    with open(args.config, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    loggers = Loggers(config.common.logs)
    loggers.update_loss({'args_out': args, 'config_out': config}, True)

    train_dataset = get_one_dataset(config.train.dataset)
    test_datasets = get_dataset(config.test.dataset)

    model = model_entry(config.common.model)
    parameters = model.set_params()
    config.train.optimizer['lr'] = config.train.lr_scheduler.base_lr
    optimizer = get_optimizer(config.train.optimizer, parameters)
    # load model params
    lowest_err, last_iter = float('inf'), -1
    if config.common.load.load:  # load model
        load_path = config.common.load.path
        print('load model from %s' % load_path)
        if not os.path.exists(load_path):
            raise AssertionError('load_path not exist')
        load_way = config.common.load.get('type', 'recover')
        if load_way == 'recover':
            print('Resume training from a previous checkpoint ...')
            lowest_err, last_iter = load_state(
                load_path, model, optimizer=optimizer)
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
