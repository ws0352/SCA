import importlib
import logging
import time
import os
import torch
import shutil
import numpy as np
import random
from skimage.io import imsave
from functools import reduce
from collections import OrderedDict

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)

LABEL_MAP = OrderedDict(
    Background=0,
    Building=1,
    Road=2,
    Water=3,
    Barren=4,
    Forest=5,
    Agricultural=6
)

def seed_torch(seed=3111):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

def import_config(cfg_path):
    cfg = importlib.import_module(name=cfg_path)
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    shutil.copy(cfg_path.replace('.', '/')+'.py', os.path.join(cfg.SNAPSHOT_DIR, 'config.py'))

    return cfg

def get_console_file_logger(name, level=logging.INFO, logdir='./log'):
    logger = logging.Logger(name)
    logger.setLevel(level=level)
    logger.handlers = []

    BASIC_FORMAT = "%(asctime)s, %(levelname)s:%(name)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    # file
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)

    # console
    fhlr = logging.FileHandler(os.path.join(logdir, str(time.time()) + '.log'))
    fhlr.setFormatter(formatter)

    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    return logger


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, cfg):
    if i_iter < cfg.PREHEAT_STEPS:
        lr = lr_warmup(cfg.LEARNING_RATE, i_iter, cfg.PREHEAT_STEPS)
    else:
        lr = lr_poly(cfg.LEARNING_RATE, i_iter, cfg.NUM_STEPS_MAX, cfg.POWER)

    optimizer.param_groups[0]['lr'] = lr

    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

    return lr


# def count_model_parameters(module, _default_logger=None):
#     cnt = 0
#     for p in module.parameters():
#         cnt += reduce(lambda x, y: x * y, list(p.shape))
#     _default_logger.info('#params: {}, {} M'.format(cnt, round(cnt / float(1e6), 3)))
#
#     return cnt
