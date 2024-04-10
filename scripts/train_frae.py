'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-03-21 15:37:34
LastEditTime: 2024-04-10 18:59:40
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/scripts/train_frae.py
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
from focal_frequency_loss import FocalFrequencyLoss

import sys
sys.path.append('/data/wxl/code')

from FreqDefense.utils.utils import DictToObject, Low_freq_substitution, addRayleigh_noise
from FreqDefense.datasets.datautils import getDataloader, getImageSize, getNormalizeParameter
from FreqDefense.models.frae import FRAE

###########################################################################
#  accelerate launch --multi_gpu --num_processes=2 scripts/train_frae.py  #
###########################################################################

# image log method
@torch.no_grad()
def log_recons_image(datasetname, name, imagelist, steps, writer):
    mean, std = getNormalizeParameter(datasetname)
    std1 = torch.tensor(std).view(1, -1, 1, 1).cuda()
    mean1 = torch.tensor(mean).view(1, -1, 1, 1).cuda()
    img_total = torch.tensor([]).cuda()
    for i, img in enumerate(imagelist):
        if img.shape[0] <=4:
            img = img[:, :, :, :]
        else:
            img = img[0:4, :, :, :]
        img = img * std1 + mean1
        img_total = torch.cat([img_total, img], dim=0).clamp(0, 1)
    img = make_grid(img_total, 4)
    writer.add_image(name, img, steps)
    writer.flush()


def train_one_epoch(model, optimizer, lpips, FFL, train_dataloader, data_config, train_config, accelerator, writer, logger, cur_epoch, low_freq_substitution=None, high_noise=None):
    model.train()
    print_steps = len(train_dataloader) // train_config.print_freq
    _, size = getImageSize(data_config.dataset_name)

    for i, (img, _) in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()
        global_steps = cur_epoch * len(train_dataloader) + i
        if train_config.f_distortion:
            if low_freq_substitution is None:
                raise Exception("low_freq_substitution is None")
            if high_noise is None:
                raise Exception("high_noise is None")
            img_f = low_freq_substitution(img)
            img_f = high_noise(img_f)
            if train_config.augment:
                # augment the image
                all_img = torch.cat([img_f, img], dim=0)
                output = model(all_img)
                correct_img = torch.cat([img, img], dim=0)
            else:
                all_img = img_f
                correct_img = img
                output = model(img_f)
        else:
            all_img = img
            correct_img = img
            output = model(img)
        
        loss = torch.zeros(1).to(all_img.device)
        
        if train_config.lpips_loss:
            loss_lpips = lpips(output, correct_img).mean()
            # check if the loss is nan
            if torch.isnan(loss_lpips).any():
                logger.error(f"Loss is nan at epoch {cur_epoch}, iteration {i}")
                raise Exception("Loss is nan")
            loss += loss_lpips
        
        # check the shape of the loss
        loss_l1 = (correct_img - output).abs().mean()
        loss += loss_l1

        if train_config.ffl_loss:
            loss_ffl = FFL(output, correct_img, dim=0)
            loss += loss_ffl

        
        accelerator.backward(loss)
        optimizer.step()

        if accelerator.is_main_process:
            if i % print_steps == 0:
                logger.info(
                    f'Epoch {cur_epoch}/{train_config.epochs}, iteration {i}/{len(train_dataloader)}, loss: {loss.item()}, l1 loss: {loss_l1.item()}, lpips loss: {loss_lpips.mean().item() if train_config.lpips_loss else 0.0}, ffl loss: {loss_ffl.mean().item() if train_config.ffl_loss else 0.0}')
                writer.add_scalar(
                    'train/l1_loss', loss_l1.item(), global_steps)
                writer.add_scalar('train/loss', loss.item(), global_steps)
                if train_config.lpips_loss:
                    writer.add_scalar('train/loss_lpips',
                                    loss_lpips.mean().item(), global_steps)
                if train_config.ffl_loss:
                    writer.add_scalar('train/loss_ffl',
                                      loss_ffl.item(), global_steps)
                if train_config.f_distortion:
                    if train_config.augment:
                        log_recons_image(data_config.dataset_name, 'train/recons', [img[0:2,:,:,:], img_f[0:2,:,:,:], output[0:2,:,:,:],output[img.shape[0]:img.shape[0]+2,:,:,:]], global_steps, writer)
                    else:
                        log_recons_image(data_config.dataset_name, 'train/recons', [img, img_f, output], global_steps, writer)
                else:
                    log_recons_image(data_config.dataset_name,
                                 'train/recons', [img, output], global_steps, writer)


def validation_one_epoch(model, lpips, FFL, val_dataloader, data_config, train_config, accelerator, writer, logger, cur_epoch, low_freq_substitution=None, high_noise=None):
    device = accelerator.device
    model.eval()
    total_loss, total_l1_loss, total_lpips_loss, total_ffl_loss, item = torch.zeros(
        5).to(device)

    with torch.no_grad():
        for _, (img, _) in tqdm(enumerate(val_dataloader)):
            if train_config.f_distortion:
                if low_freq_substitution is None:
                    raise Exception("low_freq_substitution is None")
                if high_noise is None:
                    raise Exception("high_noise is None")
                img_f = low_freq_substitution(img)
                img_f = high_noise(img_f)
                output = model(img_f)
            else:
                output = model(img)

            loss = torch.zeros(1).to(img.device)

            loss_l1 = (img - output).abs().mean()
            loss += loss_l1
            total_l1_loss += loss_l1.item() * img.shape[0]
            

            if train_config.lpips_loss:
                lpips_loss = lpips(output, img).mean()
                loss += lpips_loss
                total_lpips_loss += lpips_loss.item() * img.shape[0]
                
            if train_config.ffl_loss:
                loss_ffl = FFL(output, img, dim=0).mean()
                loss += loss_ffl
                total_ffl_loss += loss_ffl.item() * img.shape[0]
                
            total_loss += loss.item() * img.shape[0]
            item += img.shape[0]

        # gather all the loss
        total_l1_loss = accelerator.gather(total_l1_loss)
        total_loss = accelerator.gather(total_loss)
        item = accelerator.gather(item)

        # calculate the mean loss
        total_loss = total_loss.sum().item()
        total_l1_loss = total_l1_loss.sum().item()
        item = item.sum().item()

        total_loss /= item
        total_l1_loss /= item

        if train_config.ffl_loss:
            total_ffl_loss = accelerator.gather(total_ffl_loss)
            total_ffl_loss = total_ffl_loss.sum().item()
            total_ffl_loss /= item
        if train_config.lpips_loss:
            total_lpips_loss = accelerator.gather(total_lpips_loss)
            total_lpips_loss = total_lpips_loss.sum().item()
            total_lpips_loss /= item

        if accelerator.is_main_process:
            logger.info(
                f'Epoch {cur_epoch}/{train_config.epochs}, validation loss: {total_loss}, l1 loss: {total_l1_loss}, lpips loss: {total_lpips_loss if train_config.lpips_loss else 0.0 }, ffl loss: {total_ffl_loss if train_config.ffl_loss else 0.0}')
            writer.add_scalar('val/total_loss', total_loss, cur_epoch)
            writer.add_scalar('val/total_l1_loss', total_l1_loss, cur_epoch)
            if train_config.ffl_loss:
                writer.add_scalar('val/total_ffl_loss',
                                  total_ffl_loss, cur_epoch)
            if train_config.lpips_loss:
                writer.add_scalar('val/total_lpips_loss',
                              total_lpips_loss, cur_epoch)
            log_recons_image(data_config.dataset_name,
                             'val/recons', [img, img_f, output], cur_epoch, writer)
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
            experiment_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
            logger.add(os.path.join(args.result_dir,
                       f'log_resume_{experiment_id}.txt'), level='DEBUG')
            logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
        else:
            raise Exception("resume_path not found")
    else:
        # create result directory
        if not 'result_name' in args.__dict__:
            experiment_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
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
        logger.add(os.path.join(args.result_dir, 'log.txt'))
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

        # objectify config
        data_config = DictToObject(args.data_config)
        model_config = DictToObject(args.model_config)
        train_config = DictToObject(args.train_config)

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
    ch_in, resolution = getImageSize(data_config.dataset_name)
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
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=train_config.lr_decay_step, gamma=train_config.lr_decay_rate)
    
    # lpips loss
    lpips = LPIPS(net='vgg')
    
    # ffl loss
    ffl_loss = None
    if train_config.ffl_loss:
        ffl_loss = FocalFrequencyLoss(train_config.ffl_weight, alpha=1.0)

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
        logger.info(f'f_distortion is True, f_alpha: {train_config.f_alpha}, f_beta: {train_config.f_beta}, f_scale: {train_config.f_scale}')
        low_freq_substitution = Low_freq_substitution(
            resolution, resolution, ch_in, low_freq_image, data_config.batch_size, train_config.f_alpha, train_config.f_beta)
        high_noise = addRayleigh_noise(resolution, resolution, ch_in, data_config.batch_size, train_config.f_alpha, train_config.f_scale)
    # prepare model and etc with accelerator
    model, optimizer, train_dataloader, val_dataloader, lpips, ffl_loss = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lpips, ffl_loss)

    # training loop
    logger.info("Start training...")
    for epoch in range(cur_epoch, train_config.epochs):
        # replace the low frequency part of the image with the low frequency part of a random image
        if train_config.f_distortion:
            idx = random.randint(0, len(train_dataloader.dataset)-1)
            low_freq_image = train_dataloader.dataset[idx][0]
            log_recons_image(data_config.dataset_name, 'train/low_freq_image', [low_freq_image.unsqueeze(0).cuda()], epoch
                         ,writer)
            low_freq_substitution.update(low_freq_image)
            low_freq_substitution, high_noise = accelerator.prepare(low_freq_substitution, high_noise)
            train_one_epoch(model, optimizer, lpips, ffl_loss, train_dataloader,
                        data_config, train_config, accelerator, writer, logger, epoch, low_freq_substitution, high_noise)
        else:
            train_one_epoch(model, optimizer, lpips, ffl_loss, train_dataloader,
                        data_config, train_config, accelerator, writer, logger, epoch)
        scheduler.step()
        if train_config.f_distortion:
            loss = validation_one_epoch(
                model, lpips, ffl_loss, val_dataloader, data_config, train_config, accelerator, writer, logger, epoch, low_freq_substitution, high_noise)
        else:
            loss = validation_one_epoch(
                model, lpips, ffl_loss, val_dataloader, data_config, train_config, accelerator, writer, logger, epoch)
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
                'data_config': data_config.to_dict(),
                'model_config': model_config.to_dict(),
                'train_config': train_config.to_dict()
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
                        default='./scripts/config/frae/32config.yaml', help="path to config file")
    args = parser.parse_args()
    main(args)
