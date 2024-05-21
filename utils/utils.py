'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-03-25 10:36:30
LastEditTime: 2024-05-15 11:41:29
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/utils/utils.py
'''

from torchvision.utils import _log_api_usage_once
from torch import Tensor
import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)
    def to_dict(self):
        dict = {}
        for key in vars(self):
            value = getattr(self, key)
            if isinstance(value, DictToObject):
                dict[key] = value.to_dict()
            else:
                dict[key] = value
        return dict


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
        self.register_buffer('mask', mask)
        self.register_buffer('low_freq_image_fft_amplitude', low_freq_image_fft_amplitude)
        
    def update_alpha(self, alpha: float) -> None:
        self.alpha = alpha
        center = ((self.input_height-1)/2, (self.input_width-1)/2)
        max_radius = min(
            center[0], center[1], self.input_height-center[0], self.input_width-center[1])
        radius = max_radius*self.alpha
        mask = torch.zeros(self.channel, self.input_height, self.input_width)
        for i in range(self.input_height):
            for j in range(self.input_width):
                if (i-center[0])**2 + (j-center[1])**2 <= radius**2:
                    mask[:,i,j] = 1
        if mask.device != self.mask.device:
            mask = mask.to(self.mask.device)
        self.mask = mask
    
    def update(self, low_freq_image: Tensor) -> None:
        assert low_freq_image.shape[0] == self.channel and low_freq_image.shape[
            1] == self.input_height and low_freq_image.shape[2] == self.input_width, 'the shape of low_freq_image should be [3, input_height, input_width]'
        low_freq_image_fft = fft2(low_freq_image, dim=(-2, -1))
        low_freq_image_fft = fftshift(low_freq_image_fft, dim=(-2, -1))
        self.low_freq_image_fft_amplitude = torch.abs(low_freq_image_fft)
        self.low_freq_image_phase = torch.angle(low_freq_image_fft)

    # replace the low frequency part of the image with the low frequency part of a random image
    def forward(self, tensor: Tensor) -> Tensor:
        # get the amplitude and phase of the input image
        tensor_fft = fft2(tensor, dim=(-2, -1))
        tensor_fft = fftshift(tensor_fft, dim=(-2, -1))
        tensor_amplitude = torch.abs(tensor_fft)
        tensor_phase = torch.angle(tensor_fft)

        mask = self.mask.clone()
        
        # from matplotlib import pyplot as plt
        # # imshow mask
        # plt.imshow(mask[0,:,:])
        # plt.show()

        low_freq_image_fft_amplitude = self.low_freq_image_fft_amplitude.clone()
        if len(tensor.shape) > 3:
            mask = mask.unsqueeze(0).repeat_interleave(tensor.shape[0], dim=0)
            low_freq_image_fft_amplitude = low_freq_image_fft_amplitude.unsqueeze(0).repeat_interleave(tensor.shape[0], dim=0)

        tensor_amplitude = self.beta * mask * low_freq_image_fft_amplitude + \
            (torch.ones_like(mask)-mask) * tensor_amplitude
        
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
    
class addRayleigh_noise(nn.Module):
    def __init__(self, input_height, input_width, channel, batch_size, alpha=0.15, scale=1):
        super(addRayleigh_noise, self).__init__()
        _log_api_usage_once(self)
        
        self.alpha = alpha
        self.input_height = input_height
        self.input_width = input_width
        self.channel = channel
        self.batch_size = batch_size
        self.scale = scale

        center = ((input_height-1)/2, (input_width-1)/2)
        max_radius = min(
            center[0], center[1], input_height-center[0], input_width-center[1])
        radius = max_radius*alpha
        mask = torch.zeros(channel, input_height, input_width)
        for i in range(input_height):
            for j in range(input_width):
                if (i-center[0])**2 + (j-center[1])**2 >= radius**2:
                    mask[:,i,j] = 1
        self.register_buffer('mask', mask)

    def update_alpha(self, alpha: float) :
        self.alpha = alpha
        center = ((self.input_height-1)/2, (self.input_width-1)/2)
        max_radius = min(
            center[0], center[1], self.input_height-center[0], self.input_width-center[1])
        radius = max_radius*self.alpha
        mask = torch.zeros(self.channel, self.input_height, self.input_width)
        for i in range(self.input_height):
            for j in range(self.input_width):
                if (i-center[0])**2 + (j-center[1])**2 >= radius**2:
                    mask[:,i,j] = 1
    def update_scale(self, scale: float) :
        self.scale = scale


    def forward(self, tensor: Tensor) -> Tensor:
        # get the amplitude and phase of the input image
        tensor_fft = fft2(tensor, dim=(-2, -1))
        tensor_fft = fftshift(tensor_fft, dim=(-2, -1))
        tensor_amplitude = torch.abs(tensor_fft)
        tensor_phase = torch.angle(tensor_fft)
        
        # generate the noise
        noise = np.random.rayleigh(self.scale, tensor.shape)
        noise = torch.tensor(noise, dtype=torch.float32)
        noise = noise.to(tensor.device)

        if len(tensor.shape) > 3:
            mask = self.mask.clone()
            mask = mask.unsqueeze(0).repeat_interleave(tensor.shape[0], dim=0)
        
        tensor_amplitude = self.mask * noise + tensor_amplitude

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

# create a new class to mask the certain area of the image
class MaskAdder(nn.Module):
    def __init__(self):
        super(MaskAdder, self).__init__()

    def forward(self, x:Tensor, pos, size):
        assert size < min(x.shape[2], x.shape[3]), 'the size of mask should be smaller than the image'
        
        # create mask for different position
        mask = torch.ones_like(x)
        if pos == 'center':
            left_top = (x.shape[2]//2-size//2, x.shape[3]//2-size//2)
            mask[:, :, left_top[0]:left_top[0]+size, left_top[1]:left_top[1]+size] = 0
        elif pos == 'lt':
            mask[:, :, 0:size, 0:size] = 0
        elif pos == 'rt':
            mask[:, :, 0:size, x.shape[3]-size:] = 0
        elif pos == 'lb':
            mask[:, :, x.shape[2]-size:, 0:size] = 0
        elif pos == 'rb':
            mask[:, :, x.shape[2]-size:, x.shape[3]-size:] = 0
        elif pos == 'all':
            x = torch.cat([self.forward(x, 'lt', size), self.forward(x, 'rt', size), self.forward(x, 'lb', size), self.forward(x, 'rb', size), self.forward(x, 'center', size)], dim=0)
            return x
        else:
            raise ValueError('the position is not supported')
        return x * mask + torch.ones_like(x) * (1-mask)

    def __call__(self, x:Tensor, pos, size):
        return self.forward(x, pos, size)

    def __repr__(self):
        return f"maskadder(size={self.alpha})"
    
# blended other image to the image
class BlendImage(nn.Module):
    def __init__(self):
        super(BlendImage, self).__init__()

    def forward(self, x:Tensor, blend_image:Tensor, alpha=0.1):
        return (1-alpha) * x + alpha * blend_image
        

    def __call__(self, x:Tensor, blend_image:Tensor, alpha=0.1):
        return self.forward(x, blend_image, alpha)

    def __repr__(self):
        return f"blend_image(alpha={self.alpha})"
    
def test_blended():
    import sys
    sys.path.append('../')
    from datasets.datautils import getDataloader
    from torchvision.transforms import transforms as T
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from PIL import Image
    datasetname = 'cifar10'
    root = '../data'
    dataloader = getDataloader(datasetname, root, 8, 4, True)
    testdataloader = getDataloader(datasetname, root, 8, 4, False)
    print(len(dataloader.dataset), len(testdataloader.dataset))
    from tqdm import tqdm
    toPIL = T.ToPILImage()
    trans = T.Compose([
        T.Resize(32),
        T.CenterCrop(32),
        T.ToTensor()
    ])
    blend_image = trans(Image.open('../test.png'))
    dataloader.dataset.transform = trans
    for x, y in tqdm(dataloader):
        all_images = []
        all_images.append(x)
        blend = BlendImage()
        for i in range(10):
            x_blended = blend(x, blend_image, i*0.1)
            all_images.append(x_blended)
        x = make_grid(torch.cat(all_images), 8)
        plt.axis('off')
        plt.imshow(toPIL(x))
        break


def test_mask():
    import sys
    sys.path.append('../')
    from datasets.datautils import getDataloader
    from torchvision.transforms import transforms as T
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    datasetname = 'cifar10'
    root = '../data'
    dataloader = getDataloader(datasetname, root, 8, 4, True)
    testdataloader = getDataloader(datasetname, root, 8, 4, False)
    print(len(dataloader.dataset), len(testdataloader.dataset))
    from tqdm import tqdm
    toPIL = T.ToPILImage()
    trans = T.Compose([
        T.Resize(32),
        T.CenterCrop(32),
        T.ToTensor()
    ])
    dataloader.dataset.transform = trans
    for x, y in tqdm(dataloader):
        maskadder = MaskAdder()
        x_masked = maskadder(x, 'all', 4)
        x = make_grid(torch.cat([x,x_masked]), 8)
        plt.imshow(toPIL(x))
        break
        


def test_lh():
    from PIL import Image
    import torchvision.transforms as T
    import numpy as np
    import torch
    size = 256
    # load the image
    img = Image.open('/data/wxl/code/FreqDefense/test.png')
    # img = Image.open('/data/wxl/code/FreqDefense/data/20-imagenet/train/n03404251/n03404251_530.JPEG')
    # transform the image
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    denormalize = T.Compose([
        T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    img = transform(img).unsqueeze(0)
    # lw = Image.open('../data/20-imagenet/train/n02860847/n02860847_8.JPEG')
    lw = Image.open('/data/wxl/code/FreqDefense/test3.png')
    lw = transform(lw)
    alpha = 0.2
    lows = Low_freq_substitution(size, size, 3, lw, 1, alpha, 1)
    for i in range(1,10):
        scale = i*10
        high = addRayleigh_noise(size, size, 3, 1, alpha, scale)
        o = lows(img)
        o = high(o)
        # print(o.shape)
        # print(img[0,0,:,:])
        # print(o[0,0,:,:])
        # show the image from one batch
        import matplotlib.pyplot as plt
        a = denormalize(img[0,:,:,:]).permute(1,2,0)
        b = denormalize(o[0,:,:,:]).permute(1,2,0)
        print(scale)
        plt.imshow(a.clamp(0,1))
        plt.show()
        plt.imshow(b.clamp(0,1))
        plt.show()

if __name__ == '__main__':
    # test_mask()
    test_blended()