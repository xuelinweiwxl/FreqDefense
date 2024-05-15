'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-04-12 10:36:00
LastEditTime: 2024-05-15 11:06:27
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/scripts/train_trn3.py
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
from FreqDefense.models.model2 import TeacherModel, Encoder, Decoder, TRN

###########################################################################
#  accelerate launch --multi_gpu --num_processes=2 scripts/train_trn3.py  #
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
        for i, img in enumerate(imagelist):
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

# using one function instead
def process(train_config, img, model, basemodel, **kwargs):
    augmented_imgs = [img]
    correct_imgs = [img]
    ncolumns = 1

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
    
    # get intermediate output of the base model
    # model_hook is a dictionary, the key is the img size and value if the intermediate output
    model_hook = basemodel(correct_imgs)
    
    # get the output of the model
    x_rec, encoder_hook, decoder_hook = model(input_imgs)

    return ncolumns, augmented_imgs, input_imgs, x_rec, correct_imgs, model_hook, encoder_hook, decoder_hook


def train_one_epoch(basemodel, model, optimizer, train_dataloader, data_config, train_config, accelerator, writer, logger, cur_epoch, **kwargs):
    model.train()
    basemodel.to(accelerator.device)
    print_steps = len(train_dataloader) // train_config.print_freq

    for i, (img, _) in tqdm(enumerate(train_dataloader)):
        basemodel.eval()
        optimizer.zero_grad()
    
        global_steps = cur_epoch * len(train_dataloader) + i
        batch_size = img.shape[0]  
        
        ncolumns, augmented_imgs, input_imgs, x_rec, correct_imgs, model_hook, encoder_hook, decoder_hook = process(train_config, img, model, basemodel, **kwargs)

        FFL = kwargs['FFL']
        
        loss_rec = torch.tensor(0.0).to(img.device)
        loss_feature = torch.tensor(0.0).to(img.device)
        loss_img = torch.tensor(0.0).to(img.device)
        loss_dict_img = {}
        loss_dict_feature = {}
        loss_dict_rec = {}
        
        # calculate the ffl loss between the intermediate output of encoder, decoder and the base model
        for key, value in encoder_hook.items():
            if 'l1' in train_config.feature_loss:
                if 'rec' in train_config.feature_loss:
                    loss1 = torch.nn.L1Loss()(value, decoder_hook[key])
                    loss_rec += loss1
                    loss_dict_rec[key] = loss1
                loss2 = torch.nn.L1Loss()(value, model_hook[key])
                loss_feature += loss2
                loss_dict_feature[key] = loss2
            if 'ffl' in train_config.feature_loss:
                if 'rec' in train_config.feature_loss:
                    loss1 = FFL(value, decoder_hook[key])
                    loss_rec += loss1
                    loss_dict_rec[key] = loss1
                loss2 = FFL(value, model_hook[key])
                loss_feature += loss2
                loss_dict_feature[key] = loss2
        
        if 'l1' in train_config.img_loss:
            loss_img_l1 = torch.nn.L1Loss()(x_rec, correct_imgs)
            loss_img += loss_img_l1
            loss_dict_img['loss_img_l1'] = loss_img_l1
        if 'lpips' in train_config.img_loss:
            assert 'lpips' in kwargs, 'lpips is None'
            lpips = kwargs['lpips']
            loss_img_lpips = lpips(x_rec, correct_imgs).mean()
            loss_img += loss_img_lpips
            loss_dict_img['loss_img_lpips'] = loss_img_lpips
        if 'ffl' in train_config.img_loss:
            assert 'FFL' in kwargs, 'ffl_loss is None'
            FFL = kwargs['FFL']
            loss_img_ffl = FFL(x_rec, correct_imgs).mean()
            loss_img += loss_img_ffl
            loss_dict_img['loss_img_ffl'] = loss_img_ffl

        loss = loss_rec + loss_feature + loss_img

        # backward and optimize
        accelerator.backward(loss)
        optimizer.step()

        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"Parameter {name} grad is None")

        # log the loss
        if accelerator.is_main_process:
            if i % print_steps == 0:
                # global info
                loss_report = f'Epoch {cur_epoch}/{train_config.epochs}, iteration {i}/{len(train_dataloader)}, loss rec{loss_rec.item()}, loss feature:{loss_feature.item()}, loss img:{loss_img.item()}, loss:{loss.item()}'
                writer.add_scalar('train/loss', loss.item(), global_steps)
                writer.add_scalar('train/loss_rec', loss_rec.item(), global_steps)
                writer.add_scalar('train/loss_feature', loss_feature.item(), global_steps)
                writer.add_scalar('train/loss_img', loss_img.item(), global_steps)
                
                # feature loss info
                for key, value in loss_dict_feature.items():
                    writer.add_scalar(f'train_loss_feature/{key}', value.item(), global_steps)

                # rec loss info
                for key, value in loss_dict_rec.items():
                    writer.add_scalar(f'train_loss_rec/{key}', value.item(), global_steps)

                # image loss info
                for key, value in loss_dict_img.items():
                    writer.add_scalar(f'train_loss_img/{key}', value.item(), global_steps)
                
                logger.info(loss_report)

                # log the image
                if ncolumns > 1:
                    index = [ i * batch_size for i in range(ncolumns)]
                else:
                    index = range(5)
                    ncolumns = 5
                log_recons_image(data_config.dataset_name, 'train/img_rec', [augmented_imgs[index], input_imgs[index], x_rec[index], correct_imgs[index]], global_steps, writer, ncolumns)
                i = index[0]
                for key in encoder_hook.keys():
                    log_recons_image(None, f'train_feature_{key}/model', [model_hook[key][i].unsqueeze(1)], global_steps, writer, ncolumns)
                    log_recons_image(None, f'train_feature_{key}/encoder', [encoder_hook[key][i].unsqueeze(1)], global_steps, writer, ncolumns)
                    log_recons_image(None, f'train_feature_{key}/decoder', [decoder_hook[key][i].unsqueeze(1)], global_steps, writer, ncolumns)



def validation_one_epoch(basemodel, model, val_dataloader, data_config, train_config, accelerator, writer, logger, cur_epoch, **kwargs):
    device = accelerator.device
    model.eval()
    basemodel.eval()

    # total loss dict for gathering all the loss
    total_loss_dict = {}
    total_loss_dict['item'] = torch.tensor(0.0).to(device)
    total_loss_dict['total_loss_rec'] = torch.tensor(0.0).to(device)
    total_loss_dict['loss_dict_feature'] = torch.tensor(0.0).to(device)
    total_loss_dict['total_loss_img'] = torch.tensor(0.0).to(device)
    total_loss_dict['total_loss'] = torch.tensor(0.0).to(device)

    with torch.no_grad():
        for _, (img, _) in tqdm(enumerate(val_dataloader)):
            batch_size = img.shape[0]
            ncolumns, augmented_imgs, input_imgs, x_rec, correct_imgs, model_hook, encoder_hook, decoder_hook = process(train_config, img, model, basemodel, **kwargs)

            FFL = kwargs['FFL']

            # calculate the ffl loss between the intermediate output of encoder, decoder and the base model
            for key, value in encoder_hook.items():
                if 'l1' in train_config.feature_loss:
                    if 'rec' in train_config.feature_loss:
                        loss1 = torch.nn.L1Loss()(value, decoder_hook[key])
                        total_loss_dict['total_loss_rec'] += loss1 * input_imgs.shape[0]
                        total_loss_dict['total_loss'] += loss1 * input_imgs.shape[0]
                    loss2 = torch.nn.L1Loss()(value, model_hook[key])
                    total_loss_dict['loss_dict_feature'] += loss2 * input_imgs.shape[0]
                    total_loss_dict['total_loss'] += loss2 * input_imgs.shape[0]
                if 'ffl' in train_config.feature_loss:
                    if 'rec' in train_config.feature_loss:
                        loss1 = FFL(value, decoder_hook[key])
                        total_loss_dict['total_loss_rec'] += loss1 * input_imgs.shape[0]
                        total_loss_dict['total_loss'] += loss1 * input_imgs.shape[0]
                    loss2 = FFL(value, model_hook[key])
                    total_loss_dict['loss_dict_feature'] += loss2 * input_imgs.shape[0]
                    total_loss_dict['total_loss'] += loss2 * input_imgs.shape[0]

            if 'l1' in train_config.img_loss:
                if 'total_loss_img_l1' not in total_loss_dict.keys():
                    total_loss_dict['total_loss_img_l1'] = torch.tensor(0.0).to(device)
                loss_img_l1 = torch.nn.L1Loss()(x_rec, correct_imgs)
                total_loss_dict['total_loss_img_l1'] += loss_img_l1  * x_rec.shape[0]
                total_loss_dict['total_loss_img'] += loss_img_l1  * x_rec.shape[0]
                total_loss_dict['total_loss'] += loss_img_l1  * x_rec.shape[0]

            # The lpips function return N x 1 tensor, we need to sum them
            if 'lpips' in train_config.img_loss:
                assert 'lpips' in kwargs, 'lpips is None'
                if 'total_loss_img_lpips' not in total_loss_dict.keys():
                    total_loss_dict['total_loss_img_lpips'] = torch.tensor(0.0).to(device)
                lpips = kwargs['lpips']
                loss_img_lpips = lpips(x_rec, correct_imgs)
                total_loss_dict['total_loss_img_lpips'] += loss_img_lpips.sum()
                total_loss_dict['total_loss_img'] += loss_img_lpips.sum()
                total_loss_dict['total_loss'] += loss_img_lpips.sum()

            if 'ffl' in train_config.img_loss:
                assert 'FFL' in kwargs, 'ffl_loss is None'
                if 'total_loss_img_ffl' not in total_loss_dict.keys():
                    total_loss_dict['total_loss_img_ffl'] = torch.tensor(0.0).to(device)
                FFL = kwargs['FFL']
                loss_img_ffl = FFL(x_rec, correct_imgs)
                total_loss_dict['total_loss_img_ffl'] += loss_img_ffl * x_rec.shape[0]
                total_loss_dict['total_loss_img'] += loss_img_ffl * x_rec.shape[0]
                total_loss_dict['total_loss'] += loss_img_ffl * x_rec.shape[0]
            
            
            # calculate the total loss
            total_loss_dict['item'] += x_rec.shape[0]
        

        # gather all loss and get the mean loss
        for key, value in total_loss_dict.items():
            total_loss_dict[key] = accelerator.gather(value)
            if key == 'item':
                total_loss_dict[key] = total_loss_dict[key].sum()
            else:
                total_loss_dict[key] = total_loss_dict[key].sum().item() / total_loss_dict['item'].sum().item()

        if accelerator.is_main_process:
            # global info
            loss_report = f'Epoch {cur_epoch}/{train_config.epochs}, validation loss: {total_loss_dict["total_loss"]}, rec loss: {total_loss_dict["total_loss_rec"]}, img loss: {total_loss_dict["total_loss_img"]}, feature loss: {total_loss_dict["loss_dict_feature"]}'
            logger.info(loss_report)
            for key, value in total_loss_dict.items():
                writer.add_scalar(f'val/{key}', value, cur_epoch)
            # log the image
            # log the image
            if ncolumns > 1:
                index = [ i * batch_size for i in range(ncolumns)]
            else:
                index = range(5)
                ncolumns = 5
            log_recons_image(data_config.dataset_name, 'val/img_rec', [augmented_imgs[index], input_imgs[index], x_rec[index], correct_imgs[index]], cur_epoch, writer, ncolumns)
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

    # TODO: resume need to change according to save function
    if 'resume' in args.__dict__ and args.resume:
        args.result_dir = os.path.join(args.result_dir, args.resume_result)
        args.resume_result = os.path.join(args.result_dir, 'last_model.pt')
        logger.info(f"Resuming training from {args.resume_result}")
        if os.path.exists(args.resume_result):
            checkpoint = torch.load(args.resume_result)
            data_config = DictToObject(checkpoint['data_config'])
            model_config = DictToObject(checkpoint['model_config'])
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
    print(model_config.model_name)
    model_name = model_config.model_name

    # initialize model
    logger.info("Initializing model...")
    logger.info(
        f"Input channel: {ch_in}, resolution: {resolution}")
    basemodel = TeacherModel(model_name, resolution)
    model = TRN(model_name, resolution, model_config.gaussian_layer, model_config.gaussian_group_size)
    logger.info("Model initialized successfully")

    # optimizer and loss function setting
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=train_config.lr_decay_step, gamma=train_config.lr_decay_rate)
    
    # lpips loss
    if 'lpips' in train_config.img_loss or 'lpips' in train_config.feature_loss:
        lpips = LPIPS(net='vgg')
    else:
        lpips = None
    
    # ffl loss
    # if 'ffl' in train_config.img_loss or 'ffl' in train_config.amp_loss:
    #     FFL = FocalFrequencyLoss(train_config.ffl_weight, alpha=1.0)
    # else:
    #     FFL = None
    FFL = FocalFrequencyLoss(train_config.ffl_weight, alpha=1.0)

    # TODO: resume need to change according to save function
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
        train_one_epoch(basemodel, model, optimizer, train_dataloader,
                        data_config, train_config, accelerator, writer, logger, epoch, **params_dic)
        scheduler.step()
        loss = validation_one_epoch(
            basemodel, model, val_dataloader, data_config, train_config, accelerator, writer, logger, epoch, **params_dic)
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
                        default='./scripts/config/trn3/32config.yaml', help="path to config file")
    args = parser.parse_args()
    main(args)
