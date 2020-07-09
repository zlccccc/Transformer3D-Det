import logging
import os
import shutil

import torch


def create_logger(name, log_file, level=logging.INFO):
    print('create_lgoger: ', log_file)
    ensure_sub_dir(log_file)
    logger = logging.getLogger(name)
    logger.propagate = False
    formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)4s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_checkpoint(state, is_best, filename):
    ensure_sub_dir(filename)
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', 'ckpt_best_model.pth.tar')


def load_state(path, model, map_location='cpu', optimizer=None):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_location)
        own_keys = set(model.state_dict().keys())
        ckpt_keys = set(checkpoint['state_dict'].keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print('caution: missing keys from checkpoint {}: {}'.format(path, k))
        print('missing ', len(missing_keys), ' keys with ', len(own_keys), 'in total')

        model.load_state_dict(checkpoint['state_dict'], strict=False)

        lowest_error = checkpoint['lowest_error']
        step = checkpoint['step']

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> also loaded optimizer from checkpoint '{}' (step {})".format(path, step))

        return lowest_error, step


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_sub_dir(path):
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
