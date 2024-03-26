'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-03-25 10:36:30
LastEditTime: 2024-03-25 10:36:44
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/utils/utils.py
'''

from torchvision.utils import _log_api_usage_once
from torch import Tensor
import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift
import torch.nn as nn

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

# visual module
class Low_freq_substitution(nn.Module):
    # now use a fix image
    def __init__(self, input_height, input_width, channel, low_freq_image, batch_size, alpha=0.15, beta=1) -> None:
        super(Low_freq_substitution, self).__init__()
        _log_api_usage_once(self)
        # the shape of image is [C.H.W]
        # assert if not match the shape
        assert low_freq_image.shape[0] == channel and low_freq_image.shape[
            1] == input_height and low_freq_image.shape[2] == input_width, 'the shape of low_freq_image should be [3, input_height, input_width]'
        
        self.alpha = alpha
        self.beta = beta
        self.input_height = input_height
        self.input_width = input_width
        self.channel = channel
        self.batch_size = batch_size

        # prepare low frequency mask and low frequency fft
        # shape of low_freq_image_fft is [3, input_height, input_width]
        low_freq_image_fft = fft2(low_freq_image, dim=(-2, -1))
        low_freq_image_fft = fftshift(low_freq_image_fft, dim=(-2, -1))
        low_freq_image_fft_amplitude = torch.abs(low_freq_image_fft)
        low_freq_image_phase = torch.angle(low_freq_image_fft)
        
        center = ((input_height-1)/2, (input_width-1)/2)
        max_radius = min(
            center[0], center[1], input_height-center[0], input_width-center[1])
        radius = max_radius*alpha
        mask = torch.zeros(channel, input_height, input_width)
        for i in range(input_height):
            for j in range(input_width):
                if (i-center[0])**2 + (j-center[1])**2 <= radius**2:
                    mask[:,i,j] = 1
        mask = mask.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        low_freq_image_fft_amplitude = low_freq_image_fft_amplitude.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        self.register_buffer('mask', mask)
        self.register_buffer('low_freq_image_fft_amplitude', low_freq_image_fft_amplitude)
        
    
    def update(self, low_freq_image: Tensor) -> None:
        assert low_freq_image.shape[0] == self.channel and low_freq_image.shape[
            1] == self.input_height and low_freq_image.shape[2] == self.input_width, 'the shape of low_freq_image should be [3, input_height, input_width]'
        low_freq_image_fft = fft2(low_freq_image, dim=(-2, -1))
        low_freq_image_fft = fftshift(low_freq_image_fft, dim=(-2, -1))
        self.low_freq_image_fft_amplitude = torch.abs(low_freq_image_fft)
        self.low_freq_image_phase = torch.angle(low_freq_image_fft)
        self.low_freq_image_fft_amplitude = self.low_freq_image_fft_amplitude.unsqueeze(0).repeat_interleave(self.batch_size, dim=0)

    # replace the low frequency part of the image with the low frequency part of a random image
    def forward(self, tensor: Tensor) -> Tensor:
        # get the amplitude and phase of the input image
        tensor_fft = fft2(tensor, dim=(-2, -1))
        tensor_fft = fftshift(tensor_fft, dim=(-2, -1))
        tensor_amplitude = torch.abs(tensor_fft)
        tensor_phase = torch.angle(tensor_fft)
        
        tensor_amplitude = self.beta * self.mask * self.low_freq_image_fft_amplitude + \
            (torch.ones_like(self.mask)-self.mask) * tensor_amplitude

        # get the new image tensor
        tensor_fft = torch.polar(tensor_amplitude, tensor_phase)
        tensor_fft = ifftshift(tensor_fft, dim=(-2, -1))
        tensor = ifft2(tensor_fft, dim=(-2, -1))

        # *************** very important *************
        # if this return abs, the result will be wrong
        tensor = torch.real(tensor)
        return tensor

    def __call__(self, tensor: Tensor) -> Tensor:
        return self.forward(tensor)

    def __repr__(self) -> str:
        return f"low_freq_substitution(alpha={self.alpha}, beta={self.beta})"



def test():
    from PIL import Image
    import torchvision.transforms as T
    import numpy as np
    import torch

    # load the image
    img = Image.open('../data/20-imagenet/train/n01630670/n01630670_6.JPEG')
    # transform the image
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    denormalize = T.Compose([
        T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    img = transform(img).unsqueeze(0)
    lw = Image.open('../data/20-imagenet/train/n02860847/n02860847_8.JPEG')
    lw = torch.zeros(3, 256, 256)
    lows = Low_freq_substitution(256, 256, 3, lw, 1, 0.1, 1)
    lows.update(lw)
    o = lows(img)
    print(o.shape)
    print(img[0,0,:,:])
    print(o[0,0,:,:])
    # show the image from one batch
    import matplotlib.pyplot as plt
    a = denormalize(img[0,:,:,:]).permute(1,2,0)
    b = denormalize(o[0,:,:,:]).permute(1,2,0)
    plt.imshow(a.clamp(0,1))
    plt.show()
    plt.imshow(b.clamp(0,1))
    plt.show()

if __name__ == '__main__':
    test()