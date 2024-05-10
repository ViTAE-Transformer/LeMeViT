from cProfile import label
import datetime
import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion,
                           load_model, initialize_train_metrics, initialize_test_metrics, get_mean_metrics,
                           set_metrics)
import os
import logging
import json
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import numpy as np

from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from timm import utils

from models.networks import BASE_Transformer

import torchmetrics.functional as F


def get_scheduler(optimizer, opt, lr_policy):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(opt.epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        step_size = opt.epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif lr_policy == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,eta_min=1e-4,T_max=opt.epochs)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


"""
Initialize Parser and define arguments
"""
parser, metadata = get_parser_with_args()
opt = parser.parse_args()


opt.epochs = 200
opt.batch_size = 8
opt.lr_base = 0.00012
opt.num_workers = 8
opt.loss_function = "bce"

a = 2
opt.batch_size *= a
opt.lr_base *= a

opt.distributed = False

save_path = os.path.join('outputs', 'change_detection', opt.dataset + '_' + opt.backbone, opt.exp)

"""
Initialize experiments log
"""
logging.basicConfig(level=logging.INFO)


"""
Set up environment: define paths, download data, and set device
"""
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# def seed_torch(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

# seed_torch(seed=777)

if opt.dataset == 'cdd':
    opt.dataset_dir = './datasets/CDD/'
elif opt.dataset == 'levir':
	opt.dataset_dir = './datasets/Levir/'


"""
Load Model then define other aspects of the model
"""
device = utils.init_distributed_device(opt)
model = BASE_Transformer(opt, input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                            with_pos='learned', enc_depth=1, dec_depth=8).to(device)

if utils.is_primary(opt):
    logging.info('LOADING Model')

if utils.is_primary(opt):
    writer = SummaryWriter(os.path.join(save_path, 'log', f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'))


train_loader, val_loader = get_loaders(opt)

criterion = get_criterion(opt)
if opt.backbone == 'resnet':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99, weight_decay=0.0005) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    scheduler = get_scheduler(optimizer, opt, 'linear')
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr_base, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = get_scheduler(optimizer, opt, 'linear')

"""
 Set starting values
"""
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
if utils.is_primary(opt):
    logging.info('STARTING training')
total_step = -1

for epoch in range(opt.epochs):
    train_metrics = initialize_train_metrics()
    val_metrics = initialize_test_metrics()

    """
    Begin Training
    """
    model.train()
    if utils.is_primary(opt):
        logging.info('SET model mode to train!')
    batch_iter = 0
    if utils.is_primary(opt):
        tbar = tqdm(train_loader)
    else:
        tbar = train_loader
    for batch_img1, batch_img2, labels, fname in tbar:
        if utils.is_primary(opt):
            tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
        batch_iter = batch_iter+opt.batch_size
        total_step += 1
        # Set variables for training
        batch_img1 = batch_img1.float().to(device)
        batch_img2 = batch_img2.float().to(device)
        labels = labels.long().to(device)

        # Zero the gradient
        optimizer.zero_grad()

        # Get model predictions, calculate loss, backprop
        cd_preds = model(batch_img1, batch_img2)

        cd_loss = criterion(cd_preds, labels)
        loss = cd_loss
        loss.backward()
        optimizer.step()

        #cd_preds = cd_preds[-1] # BIT输出不是tuple
        _, cd_preds = torch.max(cd_preds, 1)

        # Calculate and log other batch metrics
        cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                       (labels.size()[0] * (opt.patch_size**2)))

        # cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
        #                        cd_preds.data.cpu().numpy().flatten(),
        #                        average='binary',
        #                        pos_label=1,zero_division=0)

        train_metrics = set_metrics(train_metrics,
                                    cd_loss,
                                    cd_corrects,
                                    # cd_train_report,
                                    scheduler.get_last_lr())

        # log the batch mean metrics
        mean_train_metrics = get_mean_metrics(train_metrics)
        if utils.is_primary(opt):
            for k, v in mean_train_metrics.items():
                writer.add_scalars(str(k), {'train': v}, total_step)

        # clear batch variables from memory
        del batch_img1, batch_img2, labels

    scheduler.step()
    if utils.is_primary(opt):
        logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

    """
    Begin Validation
    """
    model.eval()
    with torch.no_grad():
        for batch_img1, batch_img2, labels, fname in val_loader:
            # Set variables for training
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            labels = labels.long().to(device)

            # Get predictions and calculate loss
            cd_preds = model(batch_img1, batch_img2)

            cd_loss = criterion(cd_preds, labels)

            #cd_preds = cd_preds[-1] # BIT输出不是tuple
            _, cd_preds = torch.max(cd_preds, 1)

            # Calculate and log other batch metrics
            cd_corrects = (100 *
                           (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                           (labels.size()[0] * (opt.patch_size**2)))
            
            
            p = F.precision(cd_preds.flatten(), labels.flatten(), task="binary")
            r = F.recall(cd_preds.flatten(), labels.flatten(), task="binary")
            f = F.f1_score(cd_preds.flatten(), labels.flatten(), task="binary")
            
            cd_val_report = (p,r,f)

            # cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
            #                      cd_preds.data.cpu().numpy().flatten(),
            #                      average='binary',
            #                      pos_label=1,zero_division=0)

            val_metrics = set_metrics(val_metrics,
                                      cd_loss,
                                      cd_corrects,
                                      scheduler.get_last_lr(),
                                      cd_report=cd_val_report)

            # log the batch mean metrics
            mean_val_metrics = get_mean_metrics(val_metrics)
            if utils.is_primary(opt):
                for k, v in mean_train_metrics.items():
                    writer.add_scalars(str(k), {'val': v}, total_step)

            # clear batch variables from memory
            del batch_img1, batch_img2, labels
            
        if utils.is_primary(opt):
            logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))

        """
        Store the weights of good epochs based on validation results
        """
        if ((mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions'])
                or
                (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls'])
                or
                (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores'])):

            
            if utils.is_primary(opt):
                # Insert training and epoch information to metadata dictionary
                logging.info('updata the model')
                metadata['validation_metrics'] = mean_val_metrics

                # Save model and log
                
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # with open('./tmp/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                #     json.dump(metadata, fout)

                torch.save(model.state_dict(), save_path + '/checkpoint_epoch_'+str(epoch)+'.pth')

                # comet.log_asset(upload_metadata_file_path)
                best_metrics = mean_val_metrics

        if utils.is_primary(opt):
            print('An epoch finished.')
            
if utils.is_primary(opt):
    writer.close()  # close tensor board
    print('Done!')