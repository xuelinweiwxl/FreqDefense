'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-03-28 22:34:31
LastEditTime: 2024-04-03 16:29:26
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/models/model_utils.py
'''

import os
import torch

import sys
sys.path.append('/data/wxl/code')

from FreqDefense.datasets.datautils import getImageSize
from FreqDefense.models.frae import FRAE
from FreqDefense.utils.utils import Low_freq_substitution, addRayleigh_noise, DictToObject


# note that the result path should be the absolute path
def load_model(result_path, device, best=False):
    # check the result path
    if not os.path.exists(result_path):
        raise Exception("The result path does not exist.")
    
    # check the config file
    config_path = os.path.join(result_path, "config.yaml")
    if not os.path.exists(config_path):
        raise Exception("The config file does not exist.")
    
    if best:
        # check the best model
        if not os.path.exists(os.path.join(result_path, "best_model.pt")):
            raise Exception("The best model does not exist.")
        result_path = os.path.join(result_path, "best_model.pt")
    else:
        # check the last model
        if not os.path.exists(os.path.join(result_path, "last_model.pt")):
            raise Exception("The last model does not exist.")
        result_path = os.path.join(result_path, "last_model.pt")
    
    # load the model
    checkpoint = torch.load(result_path, map_location=device)
    train_config = DictToObject(checkpoint['train_config'])
    data_config = DictToObject(checkpoint['data_config'])
    model_config = DictToObject(checkpoint['model_config'])
    model_state = checkpoint['model']
    print("Load the best model from %s" % result_path)
    ch_in, resolution = getImageSize(data_config.dataset_name)
    print(f"Note: the input dimension is {ch_in} x {resolution} x {resolution}")
    if 'd_factor' in model_config.__dict__:
        if model_config.d_factor:
            if model_config.d_factor == 6:
                ch_muls = [1, 1, 2, 2, 4, 8]
            elif model_config.d_factor == 5:
                ch_muls = [1, 1, 2, 4, 8]
            elif model_config.d_factor == 4:
                ch_muls = [1, 2, 4, 4]
    model = FRAE(ch_in, model_config.bs_chanel,
                 resolution, ch_muls, model_config.dropout)
    # load the model state and send to the device
    model.load_state_dict(model_state)
    model = model.to(device)
    model.enable_external()
    model.eval()
    print("Load the model successfully.")
    if train_config.f_distortion:
        print('This model is trained with frequency distortion.')
        print('Loading the frequency distortion module ...')
        low_freq_image = torch.zeros(3, resolution, resolution)
        print(f'f_distortion is True, f_alpha: {train_config.f_alpha}, f_beta: {train_config.f_beta}')
        low_freq_substitution = Low_freq_substitution(
            resolution, resolution, ch_in, low_freq_image, data_config.batch_size, train_config.f_alpha, train_config.f_beta)
        high_noise = addRayleigh_noise(resolution, resolution, ch_in, data_config.batch_size, train_config.f_alpha, train_config.f_scale)
        print('Frequency distortion module loaded.')
        return [model, low_freq_substitution, high_noise]
    return [model]

# note that the result path should be the absolute path
def recover(result_path, best=False):
    # check the result path
    if not os.path.exists(result_path):
        raise Exception("The result path does not exist.")
    
    # check the config file
    config_path = os.path.join(result_path, "config.yaml")
    if not os.path.exists(config_path):
        raise Exception("The config file does not exist.")
    
    if best:
        # check the best model
        if not os.path.exists(os.path.join(result_path, "best_model.pt")):
            raise Exception("The best model does not exist.")
        result_path = os.path.join(result_path, "best_model.pt")
    else:
        # check the last model
        if not os.path.exists(os.path.join(result_path, "last_model.pt")):
            raise Exception("The last model does not exist.")
        result_path = os.path.join(result_path, "last_model.pt")
    
    # load the model
    checkpoint = torch.load(result_path)
    train_config = checkpoint['train_config']
    data_config = checkpoint['data_config']
    model_config = checkpoint['model_config']
    model_state = checkpoint['model']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Load the best model from %s" % result_path)
    state = {
                'model': model_state,
                'epoch': epoch,
                'loss': loss,
                'data_config': data_config.to_dict(),
                'model_config': model_config.to_dict(),
                'train_config': train_config.to_dict()
            }
    torch.save(state, result_path)
    print("The model is recovered successfully.")

def test():
    result_path = "/data/wxl/code/FreqDefense/results/2024_04_02_23_42"
    device = torch.device("cpu")
    model_list = load_model(result_path, device, best=True)
    if len(model_list) >1:
        model, low_freq_sub, high_noise = model_list[0], model_list[1], model_list[2]
    from torchvision import transforms as tf
    from PIL import Image
    from matplotlib import pyplot as plt
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    to_tensor = tf.Compose([
        tf.Resize((32, 32)),
        tf.CenterCrop(32),
        tf.ToTensor(),
        tf.Normalize(mean, std)
    ])
    path = '/data/wxl/code/FreqDefense/test.png'
    image = Image.open(path)
    plt.imshow(image)
    plt.show()
    test_data = to_tensor(image).unsqueeze(0)
    test_data = test_data.to(device)
    if low_freq_sub is not None:
        lw = Image.open('/data/wxl/code/FreqDefense/test2.png')
        low_freq_sub.update(to_tensor(lw))
        test_data = low_freq_sub(test_data)
    if high_noise is not None:
        test_data = high_noise(test_data)
    outputs = model(test_data)
    std = torch.tensor(std).view( 3, 1, 1)
    mean = torch.tensor(mean).view( 3, 1, 1)
    test_data = test_data.squeeze(0)
    test_data = test_data * std + mean
    test_data = test_data.detach().permute(1, 2, 0).numpy().clip(0, 1)
    plt.imshow(test_data)
    plt.show()
    for i in range(outputs.size(0)):
        tensor = outputs[i,:,:,:].detach()
        tensor = tensor.squeeze(0)
        tensor = tensor * std + mean
        tensor = tensor.detach().permute(1, 2, 0).numpy().clip(0, 1)
        plt.imshow(tensor)
        plt.show()

if __name__ == "__main__":
    test()
    # recover("/data/wxl/code/FreqDefense/results/2024_04_02_23_42", best=False)