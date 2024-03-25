'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-03-21 15:37:34
LastEditTime: 2024-03-25 15:27:47
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

from models.frae import FRAE
from datasets.datautils import getDataloader, getImgaeSize, getNormalizeParameter
from utils.utils import DictToObject

# image log method
@torch.no_grad()
def log_recons_image(datasetname, name, x, x_rec, steps, writer):
    mean, std = getNormalizeParameter(datasetname)
    std1 = torch.tensor(std).view(1, -1, 1, 1).cuda()
    mean1 = torch.tensor(mean).view(1, -1, 1, 1).cuda()
    x_rec = x_rec * std1 + mean1  # [B, C, H, W]
    x = x * std1 + mean1
    img = torch.cat([x, x_rec], dim=0).clamp(0, 1)
    img = make_grid(img, x.size(0))
    writer.add_image(name, img, steps)
    writer.flush()


def train(model, optimizer, lpips, train_dataloader, data_config, train_config, accelerator, writer, logger):
    device = accelerator.device
    model.train()
    lpips = lpips.to(device)
    
    print_steps = len(train_dataloader) // train_config.print_freq

    for epoch in range(train_config.epochs):
        logger.info(f"Epoch {epoch+1}/{train_config.epochs}")
        for i, (img, _) in tqdm(enumerate(train_dataloader)):
            img = img.to(device)
            optimizer.zero_grad()
            output = model(img)
            lpips_loss = lpips(output, img)
            # check if the loss is nan
            if torch.isnan(lpips_loss).any():
                logger.error(f"Loss is nan at epoch {epoch+1}, iteration {i+1}")
                raise Exception("Loss is nan")
            # check the shape of the loss
            loss = lpips_loss
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            global_steps = epoch * len(train_dataloader) + i
            
            if i % print_steps==0:
                logger.info(f'Epoch {epoch+1}/{train_config.epochs}, iteration {i}/{len(train_dataloader)}, loss: {loss.item()}')
                writer.add_scalar('train/loss', loss.item(), global_steps)
                writer.add_scalar('train/lpips_loss', lpips_loss.mean().item(), global_steps)
                log_recons_image(data_config.dataset_name,'train/recons', img, output, global_steps, writer)
            

def main(args):
    # logger setting
    logger.remove()
    logger.add(os.path.join(args.result_dir, 'log.txt'))
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

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
    copyfile(args.config_path, os.path.join(args.result_dir, 'config.yaml'))
    logger.info(f"Result directory created successfully: {args.result_dir}")

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
    sync_code = False
    if accelerator.state.num_processes > 1:
        num_processes = accelerator.state.num_processes
        logger.info("The number of gpus used: ", num_processes)

    # loading data
    logger.info('Initializing Dataloader...')
    train_dataloader = getDataloader(data_config.dataset_name, data_config.data_root, data_config.batch_size, data_config.num_workers, True)
    val_dataloader = getDataloader(data_config.dataset_name, data_config.data_root, data_config.batch_size, data_config.num_workers, False)
    logger.info(f'Number of training samples: {len(train_dataloader.dataset)}')
    logger.info(f'Number of validation samples: {len(val_dataloader.dataset)}')
    logger.info('Dataloader initialized successfully')

    # get model setting according to the dataset
    ch_in, resolution = getImgaeSize(data_config.dataset_name)
    if 'd_factor' in model_config.__dict__:
        if model_config.d_factor:
            if model_config.d_factor == 6:
                ch_muls = [1,1,2,2,4,8]
            elif model_config.d_factor == 5:
                ch_muls = [1,1,2,4,8]
            elif model_config.d_factor == 4:
                ch_muls = [1,2,4,4]

    # initialize model
    logger.info("Initializing model...")
    logger.info(f"Input channel: {ch_in}, resolution: {resolution}, ch_muls: {ch_muls}")
    model = FRAE(ch_in, model_config.bs_chanel, resolution, ch_muls, model_config.dropout)
    logger.info("Model initialized successfully")
    
    # prepare model and etc with accelerator
    model = accelerator.prepare(model)

    # optimizer and loss function setting 
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    lpips = LPIPS(net='vgg')

    # training loop
    logger.info("Start training...")
    train(model, optimizer, lpips, train_dataloader,data_config, train_config, accelerator, writer, logger)


# Running this script at the project root directory
if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Frequency based Backdoor Defense")
	# some examples
	parser.add_argument("--result_dir", type=str, default='./results', help="path to save outputs (ckpt, tensorboard runs)")
	parser.add_argument("--config_path", type=str, default='./scripts/config.yaml', help="path to config file")
	args = parser.parse_args()
	main(args)