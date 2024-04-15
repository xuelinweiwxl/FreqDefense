'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-04-12 10:36:00
LastEditTime: 2024-04-15 23:03:13
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/scripts/train_trn2.py
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
from torch.fft import fft2, ifft2, fftshift, ifftshift
import seaborn as sns
from matplotlib import pyplot as plt

import sys
sys.path.append('/data/wxl/code')

from FreqDefense.utils.utils import DictToObject, Low_freq_substitution, addRayleigh_noise
from FreqDefense.utils.utils import BlendImage, MaskAdder
from FreqDefense.datasets.datautils import getDataloader, getImageSize, getNormalizeParameter
from FreqDefense.models.frae import FRAE

###########################################################################
#  accelerate launch --multi_gpu --num_processes=2 scripts/train_trn2.py  #
###########################################################################

# image log method
@torch.no_grad()
def log_recons_image(datasetname, name, imagelist, steps, writer, ncolumns=4, heatmap=False):
    if not heatmap:
        if datasetname is None:
            std1 = 1
            mean1 = 0
        else:
            mean, std = getNormalizeParameter(datasetname)
            std1 = torch.tensor(std).view(1, -1, 1, 1).cuda()
            mean1 = torch.tensor(mean).view(1, -1, 1, 1).cuda()
        img_total = torch.tensor([]).cuda()
        for _, img in enumerate(imagelist):
            img = img * std1 + mean1
            img_total = torch.cat([img_total, img], dim=0).clamp(0, 1)
        img = make_grid(img_total, ncolumns)
        writer.add_image(name, img, steps)
        writer.flush()
    else:
        figure, axs = plt.subplots(len(imagelist), ncolumns, figsize=(ncolumns, len(imagelist)))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i, imgs in enumerate(imagelist):
            for j, img in enumerate(imgs):
                img = torch.mean(img, dim=0)
                img = img.cpu().numpy()
                sns.heatmap(img, cmap='viridis', cbar=False, ax=axs[i, j])
                axs[i, j].axis('off')
        plt.ioff()
        writer.add_figure(name, figure, steps)
        writer.flush()


def train_one_epoch(model, optimizer, train_dataloader, data_config, train_config, accelerator, writer, logger, cur_epoch, **kwargs):
    model.train()
    print_steps = len(train_dataloader) // train_config.print_freq

    for i, (img, _) in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()
    
        global_steps = cur_epoch * len(train_dataloader) + i
        batch_size = img.shape[0]
        ncolumns = 1

        augmented_imgs = [img]
        correct_imgs = [img]

        # mask the image
        if train_config.masked.enable:
            assert 'mask_adder' in kwargs, 'mask_adder is None'
            mask_adder = kwargs['mask_adder']
            method = train_config.masked.method
            masked_img = mask_adder(img, method, train_config.masked.size)
            augmented_imgs.append(masked_img)
            if method == 'all':
                correct_imgs.append(img.repeat(5,1,1,1))
                ncolumns += 5
            else:
                correct_imgs.append(img)
                ncolumns += 1
        
        # blend the image
        if train_config.blend:
            assert 'blend_image' in kwargs, 'blend_image is None'
            assert 'blend' in kwargs, 'blend is None'
            blend_image = kwargs['blend_image']
            blend = kwargs['blend']
            for j in range(5):
                blended_img = blend(img, blend_image, (j+1) * 0.1)
                augmented_imgs.append(blended_img)
                correct_imgs.append(img)
                ncolumns += 1
        
        # maybe add more

        # concatenate the images
        augmented_imgs = torch.cat(augmented_imgs, dim=0)
        correct_imgs = torch.cat(correct_imgs, dim=0)

        # adding distortion
        if train_config.f_distortion.enable:
            # get frequency distortion function
            assert 'low_freq_substitution' in kwargs, 'low_freq_substitution is None'
            assert 'high_noise' in kwargs, 'high_noise is None'
            low_freq_substitution = kwargs['low_freq_substitution']
            high_noise = kwargs['high_noise']

            # add distortion to the image
            input_imgs_f = low_freq_substitution(augmented_imgs)
            input_imgs_f = high_noise(input_imgs_f)
            input_imgs = input_imgs_f
        else:
            input_imgs = augmented_imgs
        
        # get fft of the input images
        input_imgs_fft = fft2(input_imgs, dim=(-2, -1))
        input_imgs_fft = fftshift(input_imgs_fft, dim=(-2, -1))
        input_imgs_amplitude = torch.abs(input_imgs_fft)
        input_imgs_phase = torch.angle(input_imgs_fft)
        input_imgs_amplitude_log = torch.log10(input_imgs_amplitude + 1.0)

        # get fft of the correct images
        correct_imgs_fft = fft2(correct_imgs, dim=(-2, -1))
        correct_imgs_fft = fftshift(correct_imgs_fft, dim=(-2, -1))
        correct_imgs_amplitude = torch.abs(correct_imgs_fft)
        correct_imgs_phase = torch.angle(correct_imgs_fft)
        correct_imgs_amplitude_log = torch.log10(correct_imgs_amplitude + 1.0)

        # get the output of the model
        output_amplitude_log = model(input_imgs_amplitude_log)

        # calculate the amplitude loss
        loss_amplitude = torch.tensor(0.0).to(input_imgs.device)
        loss_dict_amp = {}
        # TODO: maybe this can be changed, the loss are different between low and high frequency, and then add them together
        if 'l1' in train_config.amp_loss:
            loss_freq_l1 = torch.nn.L1Loss()(output_amplitude_log, correct_imgs_amplitude_log)
            loss_amplitude += loss_freq_l1
            loss_dict_amp['loss_freq_l1'] = loss_freq_l1
        if 'mse' in train_config.amp_loss:
            loss_freq_mse = torch.nn.MSELoss()(output_amplitude_log, correct_imgs_amplitude_log)
            loss_amplitude += loss_freq_mse
            loss_dict_amp['loss_freq_mse'] = loss_freq_mse
        
        # reconstruct the output image
        output_amplitude = torch.exp(output_amplitude_log) - 1.0
        output_fft = torch.polar(output_amplitude, input_imgs_phase)
        output_fft = ifftshift(output_fft, dim=(-2, -1))
        output_imgs = ifft2(output_fft, dim=(-2, -1))
        output_imgs = torch.real(output_imgs)

        # calculate the image loss
        loss_img = torch.tensor(0.0).to(input_imgs.device)
        loss_dict_img = {}
        if 'l1' in train_config.img_loss:
            loss_img_l1 = torch.nn.L1Loss()(output_imgs, correct_imgs)
            loss_img += loss_img_l1
            loss_dict_img['loss_img_l1'] = loss_img_l1
        if 'lpips' in train_config.img_loss:
            assert 'lpips' in kwargs, 'lpips is None'
            lpips = kwargs['lpips']
            loss_img_lpips = lpips(output_imgs, correct_imgs).mean()
            loss_img += loss_img_lpips
            loss_dict_img['loss_img_lpips'] = loss_img_lpips
        if 'ffl' in train_config.img_loss:
            assert 'FFL' in kwargs, 'ffl_loss is None'
            FFL = kwargs['FFL']
            loss_img_ffl = FFL(output_imgs, correct_imgs).mean()
            loss_img += loss_img_ffl
            loss_dict_img['loss_img_ffl'] = loss_img_ffl
        
        # calculate the total loss
        loss = loss_amplitude + loss_img

        # backward and optimize
        accelerator.backward(loss)
        optimizer.step()

        # log the loss
        if accelerator.is_main_process:
            if i % print_steps == 0:
                # global info
                loss_report = f'Epoch {cur_epoch}/{train_config.epochs}, iteration {i}/{len(train_dataloader)}, loss amp:{loss_amplitude.item()}, loss img:{loss_img.item()}, loss:{loss.item()}'
                writer.add_scalar('train/loss', loss.item(), global_steps)
                writer.add_scalar('train/loss_amp', loss_amplitude.item(), global_steps)
                writer.add_scalar('train/loss_img', loss_img.item(), global_steps)
                
                # amplitude loss info
                for key, value in loss_dict_amp.items():
                    writer.add_scalar(f'train_amp_loss/{key}', value.item(), global_steps)
                    loss_report += f', {key}: {value.item()}'
                
                # image loss info
                for key, value in loss_dict_img.items():
                    writer.add_scalar(f'train_img_loss/{key}', value.item(), global_steps)
                    loss_report += f', {key}: {value.item()}'
                
                logger.info(loss_report)

                # log the image
                index = [ i * batch_size for i in range(ncolumns)]
                log_recons_image(None, 'train/amp_rec', [input_imgs_amplitude_log[index], output_amplitude_log[index]], global_steps, writer, ncolumns)
                log_recons_image(None, 'train/amp_heatmap', [input_imgs_amplitude_log[index], output_amplitude_log[index]], global_steps, writer, ncolumns, heatmap=True)
                if train_config.f_distortion:
                    log_recons_image(data_config.dataset_name, 'train/img', [correct_imgs[index], input_imgs_f[index], output_imgs[index]], global_steps, writer, ncolumns)
                log_recons_image(data_config.dataset_name, 'train/img_rec', [correct_imgs[index], output_imgs[index]], global_steps, writer, ncolumns)
        
                



def validation_one_epoch(model, val_dataloader, data_config, train_config, accelerator, writer, logger, cur_epoch, **kwargs):
    device = accelerator.device
    model.eval()

    # total loss dict for gathering all the loss
    total_loss_dict = {}
    total_loss_dict['item'] = torch.tensor(0.0).to(device)
    total_loss_dict['total_loss_amplitude'] = torch.tensor(0.0).to(device)
    total_loss_dict['total_loss_img'] = torch.tensor(0.0).to(device)
    total_loss_dict['total_loss'] = torch.tensor(0.0).to(device)

    with torch.no_grad():
        for _, (img, _) in tqdm(enumerate(val_dataloader)):
            batch_size = img.shape[0]
            ncolumns = 1

            augmented_imgs = [img]
            correct_imgs = [img]

            # mask the image
            if train_config.masked.enable:
                assert 'mask_adder' in kwargs, 'mask_adder is None'
                mask_adder = kwargs['mask_adder']
                method = 'all'
                masked_img = mask_adder(img, method, train_config.masked.size)
                augmented_imgs.append(masked_img)
                if method == 'all':
                    correct_imgs.append(img.repeat(5,1,1,1))
                    ncolumns += 5
                else:
                    correct_imgs.append(img)
                    ncolumns += 1
            
            # blend the image
            if train_config.blend.enable:
                assert 'blend_image' in kwargs, 'blend_image is None'
                assert 'blend' in kwargs, 'blend is None'
                blend_image = kwargs['blend_image']
                blend = kwargs['blend']
                for j in range(5):
                    blended_img = blend(img, blend_image, (j+1) * 0.1)
                    augmented_imgs.append(blended_img)
                    correct_imgs.append(img)
                    ncolumns += 1
            
            # TODO:maybe add more

            # concatenate the images
            augmented_imgs = torch.cat(augmented_imgs, dim=0)
            correct_imgs = torch.cat(correct_imgs, dim=0)

            # adding distortion
            if train_config.f_distortion.enable:
                # get frequency distortion function
                assert 'low_freq_substitution' in kwargs, 'low_freq_substitution is None'
                assert 'high_noise' in kwargs, 'high_noise is None'
                low_freq_substitution = kwargs['low_freq_substitution']
                high_noise = kwargs['high_noise']

                # add distortion to the image
                input_imgs_f = low_freq_substitution(augmented_imgs)
                input_imgs_f = high_noise(input_imgs_f)
                input_imgs = input_imgs_f
            else:
                input_imgs = augmented_imgs
            
            # get fft of the input images
            input_imgs_fft = fft2(input_imgs, dim=(-2, -1))
            input_imgs_fft = fftshift(input_imgs_fft, dim=(-2, -1))
            input_imgs_amplitude = torch.abs(input_imgs_fft)
            input_imgs_phase = torch.angle(input_imgs_fft)
            input_imgs_amplitude_log = torch.log10(input_imgs_amplitude + 1.0)

            # get fft of the correct images
            correct_imgs_fft = fft2(correct_imgs, dim=(-2, -1))
            correct_imgs_fft = fftshift(correct_imgs_fft, dim=(-2, -1))
            correct_imgs_amplitude = torch.abs(correct_imgs_fft)
            correct_imgs_phase = torch.angle(correct_imgs_fft)
            correct_imgs_amplitude_log = torch.log10(correct_imgs_amplitude + 1.0)

            # get the output of the model
            output_amplitude_log = model(input_imgs_amplitude_log)

            # reconstruct the output image
            output_amplitude = torch.exp(output_amplitude_log) - 1.0
            output_fft = torch.polar(output_amplitude, input_imgs_phase)
            output_fft = ifftshift(output_fft, dim=(-2, -1))
            output_imgs = ifft2(output_fft, dim=(-2, -1))
            output_imgs = torch.real(output_imgs)

            # TODO: maybe this can be changed, the loss are different between low and high frequency, and then add them together
            if 'l1' in train_config.amp_loss:
                if 'total_loss_freq_l1' not in total_loss_dict.keys():
                    total_loss_dict['total_loss_freq_l1'] = torch.tensor(0.0).to(device)
                loss_freq_l1 = torch.nn.L1Loss()(output_amplitude_log, correct_imgs_amplitude_log)
                total_loss_dict['total_loss_freq_l1'] += loss_freq_l1 * output_amplitude_log.shape[0]
                total_loss_dict['total_loss_amplitude'] += loss_freq_l1 * output_amplitude_log.shape[0]
                total_loss_dict['total_loss'] += loss_freq_l1 * output_amplitude_log.shape[0]

            if 'mse' in train_config.amp_loss:
                if 'total_loss_freq_mse' not in total_loss_dict.keys():
                    total_loss_dict['total_loss_freq_mse'] = torch.tensor(0.0).to(device)
                loss_freq_mse = torch.nn.MSELoss()(output_amplitude_log, correct_imgs_amplitude_log)
                total_loss_dict['total_loss_freq_mse'] += loss_freq_mse * output_amplitude_log.shape[0]
                total_loss_dict['total_loss_amplitude'] += loss_freq_mse * output_amplitude_log.shape[0]
                total_loss_dict['total_loss'] += loss_freq_mse * output_amplitude_log.shape[0]

            if 'l1' in train_config.img_loss:
                if 'total_loss_img_l1' not in total_loss_dict.keys():
                    total_loss_dict['total_loss_img_l1'] = torch.tensor(0.0).to(device)
                loss_img_l1 = torch.nn.L1Loss()(output_imgs, correct_imgs)
                total_loss_dict['total_loss_img_l1'] += loss_img_l1  * output_amplitude_log.shape[0]
                total_loss_dict['total_loss_img'] += loss_img_l1  * output_amplitude_log.shape[0]
                total_loss_dict['total_loss'] += loss_img_l1  * output_amplitude_log.shape[0]

            # The lpips function return N x 1 tensor, we need to sum them
            if 'lpips' in train_config.img_loss:
                assert 'lpips' in kwargs, 'lpips is None'
                if 'total_loss_img_lpips' not in total_loss_dict.keys():
                    total_loss_dict['total_loss_img_lpips'] = torch.tensor(0.0).to(device)
                lpips = kwargs['lpips']
                loss_img_lpips = lpips(output_imgs, correct_imgs)
                total_loss_dict['total_loss_img_lpips'] += loss_img_lpips.sum()
                total_loss_dict['total_loss_img'] += loss_img_lpips.sum()
                total_loss_dict['total_loss'] += loss_img_lpips.sum()

            if 'ffl' in train_config.img_loss:
                assert 'FFL' in kwargs, 'ffl_loss is None'
                if 'total_loss_img_ffl' not in total_loss_dict.keys():
                    total_loss_dict['total_loss_img_ffl'] = torch.tensor(0.0).to(device)
                FFL = kwargs['FFL']
                loss_img_ffl = FFL(output_imgs, correct_imgs)
                total_loss_dict['total_loss_img_ffl'] += loss_img_ffl * output_amplitude_log.shape[0]
                total_loss_dict['total_loss_img'] += loss_img_ffl * output_amplitude_log.shape[0]
                total_loss_dict['total_loss'] += loss_img_ffl * output_amplitude_log.shape[0]
            
            
            # calculate the total loss
            total_loss_dict['item'] += output_amplitude_log.shape[0]
        

        # gather all loss and get the mean loss
        for key, value in total_loss_dict.items():
            total_loss_dict[key] = accelerator.gather(value)
            if key == 'item':
                total_loss_dict[key] = total_loss_dict[key].sum()
            else:
                total_loss_dict[key] = total_loss_dict[key].sum().item() / total_loss_dict['item'].sum().item()

        if accelerator.is_main_process:
            # global info
            loss_report = f'Epoch {cur_epoch}/{train_config.epochs}, validation loss: {total_loss_dict["total_loss"]}, amp loss: {total_loss_dict["total_loss_amplitude"]}, img loss: {total_loss_dict["total_loss_img"]}'
            logger.info(loss_report)
            for key, value in total_loss_dict.items():
                writer.add_scalar(f'val/{key}', value, cur_epoch)
            # log the image
            index = [ i * batch_size for i in range(ncolumns)]
            log_recons_image(None, 'val/amp_rec', [input_imgs_amplitude_log[index], output_amplitude_log[index]], cur_epoch, writer, ncolumns)
            log_recons_image(None, 'val/amp_heatmap', [input_imgs_amplitude_log[index], output_amplitude_log[index]], cur_epoch, writer, ncolumns, heatmap=True)
            if train_config.f_distortion:
                log_recons_image(data_config.dataset_name, 'val/img_rec', [correct_imgs[index], input_imgs_f[index], output_imgs[index]], cur_epoch, writer, ncolumns)
            else:
                log_recons_image(data_config.dataset_name, 'val/img_rec', [correct_imgs[index], output_imgs[index]], cur_epoch, writer, ncolumns)
    return total_loss_dict['total_loss']


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
    if 'lpips' in train_config.img_loss or 'lpips' in train_config.amp_loss:
        lpips = LPIPS(net='vgg')
    else:
        lpips = None
    
    # ffl loss
    if 'ffl' in train_config.img_loss or 'ffl' in train_config.amp_loss:
        FFL = FocalFrequencyLoss(train_config.ffl_weight, alpha=1.0)
    else:
        FFL = None

    if 'resume' in args.__dict__ and args.resume:
        model.load_state_dict(model_state)
        logger.info(f"Model loaded successfully from {args.resume_result}")
        cur_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
    else:
        cur_epoch = 0
        best_loss = 1e10

    if train_config.f_distortion.enable:
        # choose a random image from this batch
        idx = random.randint(0, len(train_dataloader.dataset)-1)
        low_freq_image = train_dataloader.dataset[idx][0].to(accelerator.device)
        low_freq_substitution = Low_freq_substitution(
            resolution, resolution, ch_in, low_freq_image, data_config.batch_size, train_config.f_distortion.f_alpha, train_config.f_distortion.f_beta)
        high_noise = addRayleigh_noise(resolution, resolution, ch_in, data_config.batch_size, train_config.f_distortion.f_alpha, train_config.f_distortion.f_scale)
    # prepare model and etc with accelerator
    model, optimizer, train_dataloader, val_dataloader, lpips, FFL = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lpips, FFL)

    # training loop
    logger.info("Start training...")
    for epoch in range(cur_epoch, train_config.epochs):
        # replace the low frequency part of the image with the low frequency part of a random image
        params_dic = {}
        if lpips:
            params_dic['lpips'] = lpips
        if FFL:
            params_dic['FFL'] = FFL
        if train_config.masked.enable:
            mask_adder = MaskAdder()
            params_dic['mask_adder'] = mask_adder
        if train_config.blend.enable:
            blend = BlendImage()
            params_dic['blend'] = blend
        if train_config.blend.enable:
            # choose a random image from this batch
            idx = random.randint(0, len(train_dataloader.dataset)-1)
            blended_image = train_dataloader.dataset[idx][0].to(accelerator.device)
            log_recons_image(data_config.dataset_name, 'train/blended_image', [blended_image.unsqueeze(0).cuda()], epoch
                         ,writer)
            params_dic['blend_image'] = blended_image
        if train_config.f_distortion.enable:
            idx = random.randint(0, len(train_dataloader.dataset)-1)
            low_freq_image = train_dataloader.dataset[idx][0]
            log_recons_image(data_config.dataset_name, 'train/low_freq_image', [low_freq_image.unsqueeze(0).cuda()], epoch
                         ,writer)
            low_freq_substitution.update(low_freq_image)
            low_freq_substitution, high_noise = accelerator.prepare(low_freq_substitution, high_noise)
            params_dic['low_freq_substitution'] = low_freq_substitution
            params_dic['high_noise'] = high_noise
        train_one_epoch(model, optimizer, train_dataloader,
                        data_config, train_config, accelerator, writer, logger, epoch, **params_dic)
        scheduler.step()
        loss = validation_one_epoch(
            model, val_dataloader, data_config, train_config, accelerator, writer, logger, epoch, **params_dic)
        writer.add_scalar(
            'train/lr', optimizer.param_groups[0]['lr'], epoch*len(train_dataloader))

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
if __name__ ==  "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Frequency based Backdoor Defense")
    # some examples
    parser.add_argument("--result_dir", type=str, default='./results',
                        help="path to save outputs (ckpt, tensorboard runs)")
    parser.add_argument("--config_path", type=str,
                        default='./scripts/config/trn2/32config.yaml', help="path to config file")
    args = parser.parse_args()
    main(args)
