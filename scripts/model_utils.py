'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-03-28 22:34:31
LastEditTime: 2024-03-28 23:49:00
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/scripts/model_utils.py
'''

import os
import torch

import sys
sys.path.append("..")

from datasets.datautils import getImgaeSize
from models.frae import FRAE


# note that the result path should be the absolute path
def load_model(result_path, device, best=True):
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
    data_config = checkpoint['data_config']
    model_config = checkpoint['model_config']
    model_state = checkpoint['model']
    print("Load the best model from %s" % result_path)
    ch_in, resolution = getImgaeSize(data_config.dataset_name)
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
    model.eval()
    print("Load the model successfully.")
    return model

def test():
    result_path = "/data/wxl/code/FreqDefense/results/2024_03_26_19_49_40"
    device = torch.device("cpu")
    model = load_model(result_path, device, best=True)
    from torchvision import transforms as tf
    from PIL import Image
    from matplotlib import pyplot as plt
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    to_tensor = tf.Compose([
        tf.Resize((256, 256)),
        tf.CenterCrop(256),
        tf.ToTensor(),
        tf.Normalize(mean, std)
    ])
    path = '/data/wxl/code/FreqDefense/data/20-imagenet/train/n01630670/n01630670_10341.JPEG'
    image = Image.open(path)
    test_data = to_tensor(image).unsqueeze(0)
    test_data = test_data.to(device)
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