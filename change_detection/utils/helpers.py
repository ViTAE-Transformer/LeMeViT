import os
import random
from typing import Callable

import logging
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from utils.dataloaders import (full_path_loader, full_test_loader, CDDloader, LEVIRloader)
from utils.metrics import jaccard_loss, dice_loss
from utils.losses import hybrid_loss
from models.networks import BASE_Transformer
from timm import utils
from timm.data.distributed_sampler import OrderedDistributedSampler

logging.basicConfig(level=logging.INFO)

def initialize_train_metrics():
    """Generates a dictionary of metrics with metrics as keys
       and empty lists as values

    Returns
    -------
    dict
        a dictionary of metrics

    """
    metrics = {
        'cd_losses': [],
        'cd_corrects': [],
        # 'cd_precisions': [],
        # 'cd_recalls': [],
        # 'cd_f1scores': [],
        'learning_rate': [],
    }

    return metrics

def initialize_test_metrics():
    """Generates a dictionary of metrics with metrics as keys
       and empty lists as values

    Returns
    -------
    dict
        a dictionary of metrics

    """
    metrics = {
        'cd_losses': [],
        'cd_corrects': [],
        'cd_precisions': [],
        'cd_recalls': [],
        'cd_f1scores': [],
        'learning_rate': [],
    }

    return metrics

def get_mean_metrics(metric_dict):
    """takes a dictionary of lists for metrics and returns dict of mean values

    Parameters
    ----------
    metric_dict : dict
        A dictionary of metrics

    Returns
    -------
    dict
        dict of floats that reflect mean metric value

    """
    return {k: np.mean(v) for k, v in metric_dict.items()}


def set_metrics(metric_dict, cd_loss, cd_corrects, lr, cd_report=None):
    """Updates metric dict with batch metrics

    Parameters
    ----------
    metric_dict : dict
        dict of metrics
    cd_loss : dict(?)
        loss value
    cd_corrects : dict(?)
        number of correct results (to generate accuracy
    cd_report : list
        precision, recall, f1 values

    Returns
    -------
    dict
        dict of  updated metrics


    """
    metric_dict['cd_losses'].append(cd_loss.item())
    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['learning_rate'].append(lr)
    if cd_report != None:
        metric_dict['cd_precisions'].append(cd_report[0].item())
        metric_dict['cd_recalls'].append(cd_report[1].item())
        metric_dict['cd_f1scores'].append(cd_report[2].item())


    return metric_dict

def set_test_metrics(metric_dict, cd_corrects, cd_report):

    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])

    return metric_dict

def _worker_init(worker_id, worker_seeding='all'):
    os.sched_setaffinity(0, range(os.cpu_count())) 
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))

def get_loaders(opt):

    if utils.is_primary(opt):
        logging.info('STARTING Dataset Creation')

    train_full_load, val_full_load = full_path_loader(opt.dataset_dir)

    if opt.dataset == 'cdd':

        train_dataset = CDDloader(opt, train_full_load, flag = 'trn', aug=opt.augmentation)
        val_dataset = CDDloader(opt, val_full_load, flag='val',  aug=False)
    
    elif opt.dataset == 'levir':
        train_dataset = LEVIRloader(opt, train_full_load, flag = 'trn', aug=opt.augmentation)
        val_dataset = LEVIRloader(opt, val_full_load, flag='val',  aug=False)
    
    if utils.is_primary(opt):
        logging.info('STARTING Dataloading')

    if opt.distributed == True:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=opt.batch_size,
                                                sampler=train_sampler,
                                                num_workers=opt.num_workers,
                                                worker_init_fn=_worker_init,
                                                pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=opt.batch_size,
                                                sampler=val_sampler,
                                                num_workers=opt.num_workers,
                                                worker_init_fn=_worker_init,
                                                pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=True,
                                                num_workers=opt.num_workers,
                                                worker_init_fn=_worker_init,
                                                pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=False,
                                                num_workers=opt.num_workers,
                                                worker_init_fn=_worker_init,
                                                pin_memory=True)
    return train_loader, val_loader

def get_test_loaders(opt, batch_size=None):

    if not batch_size:
        batch_size = opt.batch_size
    logging.info('STARTING Dataset Creation')

    test_full_load = full_test_loader(opt.dataset_dir)

    if opt.dataset == 'cdd':

        test_dataset = CDDloader(opt, test_full_load, flag = 'tes', aug=False)
        
    elif opt.dataset == 'levir':

        test_dataset = LEVIRloader(opt, test_full_load, flag = 'tes', aug=False)
    

    logging.info('STARTING Dataloading')


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers,
                                             worker_init_fn=_worker_init,
                                             pin_memory=True)
    return test_loader


def get_criterion(opt):
    """get the user selected loss function

    Parameters
    ----------
    opt : dict
        Dictionary of options/flags

    Returns
    -------
    method
        loss function

    """
    if opt.loss_function == 'hybrid':
        criterion = hybrid_loss
    if opt.loss_function == 'bce':
        criterion = nn.CrossEntropyLoss()
    if opt.loss_function == 'dice':
        criterion = dice_loss
    if opt.loss_function == 'jaccard':
        criterion = jaccard_loss

    return criterion


def load_model(opt, device):
    """Load the model

    Parameters
    ----------
    opt : dict
        User specified flags/options
    device : string
        device on which to train model

    """
    # device_ids = list(range(opt.num_gpus))
    #model = SNUNet_ECAM(opt, opt.num_channel, 2).to(device)
    # model = nn.DataParallel(model, device_ids=device_ids)

    model = BASE_Transformer(opt, input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8).to(device)

    return model
