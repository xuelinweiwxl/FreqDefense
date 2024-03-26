'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-03-21 15:37:34
LastEditTime: 2024-03-25 19:04:35
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/scripts/train.py
'''

import os
import yaml
import random
import time
import numpy as np
import torch
import datetime
from shutil import copyfile
from loguru import logger
from tqdm import tqdm
from lpips import LPIPS
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid


import sys
sys.path.append(".")
sys.path.append("..")

from utils.utils import DictToObject, Low_freq_substitution
from datasets.datautils import getDataloader, getImgaeSize, getNormalizeParameter
from models.frae import FRAE

######################################################################
#  accelerate launch --multi_gpu --num_processes=2 scripts/train.py  #
######################################################################


# image log method
@torch.no_grad()
def log_recons_image(datasetname, name, x, x_rec, steps, writer):
    x = x[0:4, :, :, :]
    x_rec = x_rec[0:4, :, :, :]
    mean, std = getNormalizeParameter(datasetname)
    std1 = torch.tensor(std).view(1, -1, 1, 1).cuda()
    mean1 = torch.tensor(mean).view(1, -1, 1, 1).cuda()
    x_rec = x_rec * std1 + mean1  # [B, C, H, W]
    x = x * std1 + mean1
    img = torch.cat([x, x_rec], dim=0).clamp(0, 1)
    img = make_grid(img, x.size(0))
    writer.add_image(name, img, steps)
    writer.flush()


def train_one_epoch(model, optimizer, lpips, train_dataloader, data_config, train_config, accelerator, writer, logger, cur_epoch, low_freq_substitution=None):
    model.train()
    print_steps = len(train_dataloader) // train_config.print_freq
    _, size = getImgaeSize(data_config.dataset_name)

    for i, (img, _) in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()
        global_steps = cur_epoch * len(train_dataloader) + i
        if train_config.f_distortion:
            if low_freq_substitution is None:
                raise Exception("low_freq_substitution is None")
            img_f = low_freq_substitution(img)
            if i % print_steps == 0:
                log_recons_image(data_config.dataset_name, 'train/f_distortion', img, img_f, global_steps, writer)
            img = img_f
        output = model(img)
        lpips_loss = lpips(output, img).mean()
        # check if the loss is nan
        if torch.isnan(lpips_loss).any():
            logger.error(f"Loss is nan at epoch {cur_epoch}, iteration {i}")
            raise Exception("Loss is nan")
        # check the shape of the loss
        loss_l1 = (img - output).abs().mean()
        loss = lpips_loss + loss_l1
        accelerator.backward(loss)
        optimizer.step()

        if accelerator.is_main_process:
            if i % print_steps == 0:
                logger.info(
                    f'Epoch {cur_epoch}/{train_config.epochs}, iteration {i}/{len(train_dataloader)}, loss: {loss.item()}, l1 loss: {loss_l1.item()}, lpips loss: {lpips_loss.mean().item()}')
                writer.add_scalar(
                    'train/l1_loss', loss_l1.item(), global_steps)
                writer.add_scalar('train/loss', loss.item(), global_steps)
                writer.add_scalar('train/lpips_loss',
                                  lpips_loss.mean().item(), global_steps)
                log_recons_image(data_config.dataset_name,
                                 'train/recons', img, output, global_steps, writer)


def validation_one_epoch(model, lpips, val_dataloader, data_config, train_config, accelerator, writer, logger, cur_epoch):
    device = accelerator.device
    model.eval()
    lpips = lpips.to(device)
    total_loss, total_l1_loss, total_lpips_loss, item = torch.zeros(
        4).to(device)

    with torch.no_grad():
        for _, (img, _) in tqdm(enumerate(val_dataloader)):
            img = img.to(device)
            output = model(img)
            lpips_loss = lpips(output, img).mean()
            loss_l1 = (img - output).abs().mean()
            loss = lpips_loss + loss_l1
            total_loss += loss.item() * img.shape[0]
            total_l1_loss += loss_l1.item() * img.shape[0]
            total_lpips_loss += lpips_loss.item() * img.shape[0]
            item += img.shape[0]
        # gather all the loss
        total_loss = accelerator.gather(total_loss)
        total_l1_loss = accelerator.gather(total_l1_loss)
        total_lpips_loss = accelerator.gather(total_lpips_loss)
        item = accelerator.gather(item)

        # calculate the mean loss
        total_loss = total_loss.mean().item()
        total_l1_loss = total_l1_loss.mean().item()
        total_lpips_loss = total_lpips_loss.mean().item()
        item = item.sum().item()

        total_loss /= item
        total_l1_loss /= item
        total_lpips_loss /= item

        if accelerator.is_main_process:
            logger.info(
                f'Epoch {cur_epoch}/{train_config.epochs}, validation loss: {total_loss}, l1 loss: {total_l1_loss}, lpips loss: {total_lpips_loss}')
            writer.add_scalar('val/total_loss', total_loss, cur_epoch)
            writer.add_scalar('val/total_l1_loss', total_l1_loss, cur_epoch)
            writer.add_scalar('val/total_lpips_loss',
                              total_lpips_loss, cur_epoch)
            log_recons_image(data_config.dataset_name,
                             'val/recons', img, output, cur_epoch, writer)
    return total_loss


def main(args):

    # config file load
    logger.info(f"Loading config file from {args.config_path}")
    if "config_path" in args.__dict__:
        with open(args.config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            for k, v in config.items():
                setattr(args, k, v)
    else:
        raise Exception("config_path not found")
    logger.info("Config file loaded successfully")

    if 'resume' in args.__dict__ and args.resume:
        args.result_dir = os.path.join(args.result_dir, args.resume_result)
        args.resume_result = os.path.join(args.result_dir, 'last_model.pt')
        logger.info(f"Resuming training from {args.resume_result}")
        if os.path.exists(args.resume_result):
            checkpoint = torch.load(args.resume_result)
            data_config = checkpoint['data_config']
            model_config = checkpoint['model_config']
            train_config = DictToObject(args.train_config)
            model_state = checkpoint['model']
            logger.info(f"{args.resume_result} loaded successfully")
            # logger setting
            logger.remove()
            experiment_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            logger.add(os.path.join(args.result_dir,
                       f'log_resume_{experiment_id}.txt'), level='DEBUG')
            logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
        else:
            raise Exception("resume_path not found")
    else:
        # create result directory
        if not 'result_name' in args.__dict__:
            experiment_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            args.result_dir = os.path.join(args.result_dir, experiment_id)
            if not os.path.exists(args.result_dir):
                os.makedirs(args.result_dir)
        else:
            experiment_id = args.result_name
            args.result_dir = os.path.join(args.result_dir, experiment_id)
            if not os.path.exists(args.result_dir):
                os.makedirs(args.result_dir)
            else:
                raise Exception("result_dir exists")
        copyfile(args.config_path, os.path.join(
            args.result_dir, 'config.yaml'))
        logger.info(
            f"Result directory created successfully: {args.result_dir}")

        # logger setting
        logger.remove()
        logger.add(os.path.join(args.result_dir, 'log.txt'), level='DEBUG')
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

        # objectify config
        data_config = DictToObject(args.data_config)
        model_config = DictToObject(args.model_config)
        train_config = DictToObject(args.train_config)

    # set random seed
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    else:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    logger.info(f"Random seed has been setted: {args.random_seed}")

    # distrubuted training and training device setting
    if 'distributed' in args.__dict__ and args.distributed:
        logger.info("Using DistributedDataParallel training mode.")
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    else:
        logger.info("Using normal training mode")
        accelerator = Accelerator()
    logger.info(f"Using device: {accelerator.device}")

    # initialize tensorboard writer
    logger.info(f'Initializing Tensorboard...')
    log_path = os.path.join(args.result_dir, "runs")
    writer = SummaryWriter(log_path, comment="comments")

    # multi gpu setting
    if accelerator.state.num_processes > 1:
        num_processes = accelerator.state.num_processes
        logger.info("The number of gpus used: ", num_processes)

    # loading data
    logger.info('Initializing Dataloader...')
    train_dataloader = getDataloader(
        data_config.dataset_name, data_config.data_root, data_config.batch_size, data_config.num_workers, True)
    val_dataloader = getDataloader(data_config.dataset_name, data_config.data_root,
                                   data_config.batch_size, data_config.num_workers, False)
    logger.info(f'Number of training samples: {len(train_dataloader.dataset)}')
    logger.info(f'Number of validation samples: {len(val_dataloader.dataset)}')
    logger.info('Dataloader initialized successfully')

    # get model setting according to the dataset
    ch_in, resolution = getImgaeSize(data_config.dataset_name)
    if 'd_factor' in model_config.__dict__:
        if model_config.d_factor:
            if model_config.d_factor == 6:
                ch_muls = [1, 1, 2, 2, 4, 8]
            elif model_config.d_factor == 5:
                ch_muls = [1, 1, 2, 4, 8]
            elif model_config.d_factor == 4:
                ch_muls = [1, 2, 4, 4]

    # initialize model
    logger.info("Initializing model...")
    logger.info(
        f"Input channel: {ch_in}, resolution: {resolution}, ch_muls: {ch_muls}")
    model = FRAE(ch_in, model_config.bs_chanel,
                 resolution, ch_muls, model_config.dropout)
    logger.info("Model initialized successfully")

    # optimizer and loss function setting
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    lpips = LPIPS(net='vgg')
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=train_config.lr_decay_step, gamma=train_config.lr_decay_rate)

    if 'resume' in args.__dict__ and args.resume:
        model.load_state_dict(model_state)
        logger.info(f"Model loaded successfully from {args.resume_result}")
        cur_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
    else:
        cur_epoch = 0
        best_loss = 1e10

    if train_config.f_distortion:
        # choose a random image from this batch
        idx = random.randint(0, len(train_dataloader.dataset)-1)
        low_freq_image = train_dataloader.dataset[idx][0].to(accelerator.device)
        low_freq_substitution = Low_freq_substitution(
            resolution, resolution, low_freq_image, train_config.f_alpha, train_config.f_beta)
    else:
        low_freq_substitution = None

    # prepare model and etc with accelerator
    model, optimizer, train_dataloader, val_dataloader, lpips = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lpips)

    # training loop
    logger.info("Start training...")
    for epoch in range(cur_epoch, train_config.epochs):
        # replace the low frequency part of the image with the low frequency part of a random image
        if train_config.f_distortion:
            idx = random.randint(0, len(train_dataloader.dataset)-1)
            low_freq_image = train_dataloader.dataset[idx][0]
            low_freq_substitution.update(low_freq_image)
            low_freq_substitution = accelerator.prepare(low_freq_substitution)
            train_one_epoch(model, optimizer, lpips, train_dataloader,
                        data_config, train_config, accelerator, writer, logger, epoch, low_freq_substitution)
        else:
            train_one_epoch(model, optimizer, lpips, train_dataloader,
                        data_config, train_config, accelerator, writer, logger, epoch)
        scheduler.step()
        loss = validation_one_epoch(
            model, lpips, val_dataloader, data_config, train_config, accelerator, writer, logger, epoch)
        writer.add_scalar(
            'train/lr', optimizer.param_groups[0]['lr'], epoch*len(train_dataloader))
        writer.add_scalar(
            'train/f_alpha', train_config.f_alpha, epoch*len(train_dataloader))
        writer.add_scalar(
            'train/f_beta', train_config.f_beta, epoch*len(train_dataloader))

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state = {
                'model': accelerator.unwrap_model(model).state_dict(),
                'epoch': epoch,
                'loss': loss,
                'data_config': data_config,
                'model_config': model_config,
                'train_config': train_config
            }
            if loss < best_loss:
                best_loss = loss
                logger.info(f"Best model found at epoch {epoch}, saving...")
                torch.save(state, os.path.join(
                    args.result_dir, 'best_model.pt'))
            logger.info(f"Saving model at epoch {epoch}...")
            torch.save(state, os.path.join(args.result_dir, f'last_model.pt'))
            logger.info(f"Model saved successfully")


# Running this script at the project root directory
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Frequency based Backdoor Defense")
    # some examples
    parser.add_argument("--result_dir", type=str, default='./results',
                        help="path to save outputs (ckpt, tensorboard runs)")
    parser.add_argument("--config_path", type=str,
                        default='./scripts/config.yaml', help="path to config file")
    args = parser.parse_args()
    main(args)
