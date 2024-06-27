'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-04-18 15:21:17
LastEditTime: 2024-05-07 16:24:58
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/models/model2.py
'''

import torch
from torchvision import models
from torch import nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import functools
from typing import List, Dict
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift, ifftshift
'''
description: the encoder and teacher model have same basic structure,
    there are still some differences between them
    1. The encoder don't have the last layers of the basic model
    2. We need to add some gaussian group filters to the encoder
    3. We need to compare the outputs of middle layers of the encoder and teacher model
    So we can create a father class for this two models to reduce the duplicate code.
'''

# HOOK_SIZE = {
#     224: [56, 28, 14],
#     256: [64, 32, 16],
#     32: [16, 8]
#     # 32: [16, 8, 4]
# }

# EMBEDING_RES = {
#     224: 14,
#     256: 8,
#     32: 4
# }

HOOK_SIZE = {
    224: [56, 28],
    256: [64, 32, 16],
    32: [16, 8]
    # 32: [16, 8, 4]
}

EMBEDING_RES = {
    224: 28,
    256: 8,
    32: 4
}

EMBEDING_CH = {
    32: {2:512, 4: 256, 8: 128, 16: 64},
    224: {7:512, 14: 256, 28: 128, 56: 64},
}

# downsample module
class Downsample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

# upsample module
class Upsample(nn.Module):
    '''
    description: upsampling module using nearest interpolation
    '''

    def __init__(self,
                 channel: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


# Resnet module
class ResBlock(nn.Module):
    '''
    description: residual block with group normalization and SiLU activation
    '''

    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 dropout: float = 0.0):

        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, ch_in),
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        )
        if ch_in != ch_out:
            self.shotcut = nn.Conv2d(
                ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        else:
            self.shotcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.shotcut(x) + self.block(x)

# Convblock module
class ConvBlock(nn.Module):
    '''
    description: convolutional block with group normalization and SiLU activation
    '''

    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

# BaseModel module
class BaseModel(nn.Module):
    '''
    description: base model for image classification
    '''

    def __init__(self,
                 model_name: str,
                 pretrained: bool = True,
                 num_classes: int = 1000,
                 input_resolution: int = 224) -> None:
        super().__init__()
        
        self.model_name = model_name
        self.pretrained = pretrained

        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained, num_classes=num_classes)
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained, num_classes=num_classes)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained, num_classes=num_classes)
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained, num_classes=num_classes)
        elif model_name == 'vgg16':
            self.model = models.vgg16_bn(pretrained, num_classes=num_classes)
        elif model_name == 'vgg19':
            self.model = models.vgg19_bn(pretrained, num_classes=num_classes)
        elif model_name == 'vitb16':
            self.model = models.vit_b_16(pretrained, num_classes=num_classes)
        elif model_name == 'convnext':
            self.model = models.convnext_tiny(pretrained, num_classes=num_classes)
        else:
            raise Exception("The model name is not supported.")
        
        self.hook_result = {}
        self.hook_channel = {}
        self.hook_modules = {}
        
        # first: determine the size we want to hook and compare
        self.hook_size = HOOK_SIZE[input_resolution]
        
        # second: get filter size of the hooked layers
        if 'resnet' in self.model_name:
            self.base_ch = self.model.conv1.out_channels
            layer_count = int(self.model_name[6:])
            if layer_count >= 50:
                self.max_ch = self.model.layer4[-1].conv3.out_channels
                for i, size in enumerate(self.hook_size):
                    layer_name = f'layer{i+1}'
                    self.hook_modules[size] = self.model.__getattr__(layer_name)[-1].conv3
                    self.hook_channel[size] = self.model.__getattr__(layer_name)[-1].conv3.out_channels
            else:
                self.max_ch = self.model.layer4[-1].conv2.out_channels
                for i, size in enumerate(self.hook_size):
                    layer_name = f'layer{i+1}'
                    self.hook_modules[size] = self.model.__getattr__(layer_name)[-1].conv2
                    self.hook_channel[size] = self.model.__getattr__(layer_name)[-1].conv2.out_channels
            if input_resolution == 32:
                self.model.maxpool = nn.Identity()
                self.zsize = self.max_ch * (input_resolution // 16) * (input_resolution // 16)
            else:
                self.zsize = self.max_ch * (input_resolution // 32) * (input_resolution // 32)
        else:
            raise Exception("The model name is not supported.")

    def hook_fn(self,
                module: nn.Module,
                input: Tensor,
                output: Tensor,
                size: int) -> None:
        self.hook_result[size] = output
    
    def forward(self, x: Tensor) -> None:
        pass

    def get_hook_result(self):
        result = self.hook_result
        self.hook_result = {}
        return result

    def __repr__(self):
        return str(self.model)
        return f'model {self.model_name} with {self.model.fc.out_features} classes'

class TeacherModel(BaseModel):
    '''
    description: teacher model for intermediate feature extraction
    feature:
    1. hook the intermediate feature maps
    2. return the feature maps without the final output
    '''
    def __init__(self, model_name: str, input_resolution: int = 224) -> None:
        super().__init__(model_name, True, 1000, input_resolution)
        # register hook
        self.hook_handles = []
        for size, module in self.hook_modules.items():
            hook_with_size = functools.partial(self.hook_fn, size=size)
            self.hook_handles.append(module.register_forward_hook(hook_with_size))

    def remove_hooks(self):
        # remove hook
        for handle in self.hook_handles:
            handle.remove()


    def forward(self, x: Tensor) -> Tensor:
        # forward to get intermediate feature maps
        x = self.model(x)
    
        return self.get_hook_result()

# Gaussian Group Filter
class GuassianFilter(nn.Module):
    '''
    description: gaussian group filter for the encoder intermediate layers
    params:
        1. kernel_size: the size of the gaussian kernel
        2. channel: the number of channels
    feature:
        1. each channel of the feature map will pass through multiple gaussian filters
            and add the results together
        2. for one gaussian filter, the sigma is a single value,
            which means all channel share the same sigma
    '''
    def __init__(self,
                kernel_size: int,
                channel: int) -> None:
        super(GuassianFilter, self).__init__()
        self.kernel_size = kernel_size
        self.channel = channel
        self.sigma = nn.Parameter(torch.randn(1))

    def get_weight(self,device):
        n = self.kernel_size // 2
        kernel_1d = torch.exp(-(torch.arange(-n, n + 1).to(device) ** 2) / (2 * self.sigma ** 2)).to(device)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_1d = kernel_1d.unsqueeze(0)
        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        kernel = kernel_2d.unsqueeze(0).repeat(self.channel,1, 1, 1)
        return kernel

    def forward(self, x: Tensor) -> Tensor:
        # TODO: check whether to regulate the sigma, maybe we can use the sigmoid function or relu function
        self.kernel = self.get_weight(x.device)
        x = nn.ReflectionPad2d(self.kernel_size // 2)(x)
        x = F.conv2d(x, weight=self.kernel, groups=self.channel)
        return x
    
# # TODO: Maybe gausssian filter is not a good idea, try to build a new module to replace it
# # combine the high pass filter and low pass filter as group filter for the encoder
# class BandPassFilter(nn.Module):
#     '''
#     description: A band pass filter for the encoder intermediate layers
#     params:
#         1. kernel_size: the size of the gaussian kernel
#         2. channel: the number of channels
#         3. range: the range of the band pass filter
#     feature:
#         1. First, we need to build the 
#     '''
#     def __init__(self,
#                 kernel_size: int,
#                 channel: int) -> None:
#         super(BandPassFilter, self).__init__()
#         self.kernel_size = kernel_size
#         self.channel = channel
#         # evert channel has a range, the maximum value is 1 and the minimum value is 0
#         self.range = nn.Parameter(torch.randn(channel))


#     def get_filter_matrix(self,device):
        

#     def forward(self, x: Tensor) -> Tensor:
#         x_fft = fft2(x, dim=(-2, -1))
#         x_fft = fftshift(x_fft, dim=(-2, -1))
#         x_ampli
#         x = nn.ReflectionPad2d(self.kernel_size // 2)(x)
#         x = F.conv2d(x, weight=self.kernel, groups=self.channel)
#         return x

# Laplacian Filter
class LaplacianFilter(nn.Module):
    '''
    description: Laplacian filter for the encoder intermediate layers
    params:
        1. kernel_size: the size of the gaussian kernel
        2. channel: the number of channels
    feature:
        1. each channel of the feature map will pass through the laplacian filter
        2. the filter is a 3 * 3 filter
    '''
    def __init__(self,
                kernel_size: int,
                channel: int) -> None:
        super(LaplacianFilter, self).__init__()
        # self.kernel_size = kernel_size
        # using 3 * 3 kernel can get more detail about the edge of the image
        self.kernel_size = 3
        self.channel = channel
        self.kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).float()
        self.kernel = self.kernel / 4
        self.kernel = self.kernel.unsqueeze(0).repeat(self.channel, 1, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        self.kernel = self.kernel.to(x.device)
        x = nn.ReflectionPad2d(self.kernel_size // 2)(x)
        x = F.conv2d(x, weight=self.kernel, groups=self.channel)
        return x

# Encoder module
'''
There are three structure need to be filled
1. different resolution
2. different channel number
3. different layer number
first we only consider resnet18 and compare with it
for resolution 32:
    1. 16 x 16 x 64
    2. 8 x 8 x 128
    3. 4 x 4 x 256
    4. 2 x 2 x 512
for resolution 224:
    1. 56 x 56 x 64
    2. 28 x 28 x 128
    3. 14 x 14 x 256
    4. 7 x 7 x 512
for resolution 256:
    1. 64 x 64 x 64
    2. 32 x 32 x 128
    3. 16 x 16 x 256
    4. 8 x 8 x 512
'''
class Encoder(nn.Module):
    '''
    description: using the same structure as the teacher model
    feature:
    1. add gaussian group filters between layers
    2. add the hook function to save the intermediate feature maps
    '''
    def __init__(self, model_name:str, resolution:int,
                 use_res:bool=True,
                 gaussian_layer:bool=True,
                 gaussian_group_size:int=3) -> None:
        
        self.gaussian_group_size = gaussian_group_size
        self.gaussian_layer = gaussian_layer

        super(Encoder, self).__init__()

        if resolution in EMBEDING_RES.keys():
            self.b_res = EMBEDING_RES[resolution]
        else:
            raise Exception(
                "The resolution is not supported yet, please check it.")

        self.hook_size = HOOK_SIZE[resolution]
        if 'resnet' in model_name:
            if int(model_name[6:]) <= 50:
                self.base_ch = 64
                self.max_ch = EMBEDING_CH[resolution][self.b_res]
            else:
                raise Exception("The model name is not supported.")
        else:
            raise Exception("The model name is not supported.")
        
        self.conv_in = nn.Conv2d(3, self.base_ch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.GroupNorm(32, self.base_ch)
        self.relu = nn.ReLU()
        if resolution != 32:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            cur_res = resolution // 2
        else:
            self.maxpool = nn.Identity()
            cur_res = resolution
        cur_ch = self.base_ch
        self.layer_count = 1
        self.hook_modules = {}
        self.hook_channel = {}
        
        while cur_res != self.b_res or cur_ch != self.max_ch:
            blocks = []
            if self.layer_count == 1:
                b_in = cur_ch
                b_out = cur_ch
            else:
                if cur_ch < self.max_ch:
                    b_in = cur_ch
                    b_out = cur_ch * 2
                else:
                    b_in = cur_ch
                    b_out = cur_ch
            if use_res:
                blocks.append(ResBlock(b_in, b_out))
            else:
                blocks.append(ConvBlock(b_in, b_out))
            
            if cur_res != self.b_res:
                blocks.append(Downsample(b_out))
                cur_res = cur_res // 2  
            else:
                cur_res = cur_res
            layer = nn.Sequential(*blocks)
            self.register_module(f'layer{self.layer_count}', layer)
            cur_ch = b_out
            if cur_res in self.hook_size:
                self.hook_modules[layer] = cur_res
                self.hook_channel[cur_res] = cur_ch
            self.layer_count += 1

        self.zsize = cur_ch * cur_res * cur_res

        if self.gaussian_layer:
            if resolution == 32:
                kernel_size = 3
            else:
                kernel_size = 5
            for size in self.hook_size:
                for i in range(gaussian_group_size):
                    self.register_module(f'gaussian{size}_{i}', GuassianFilter(kernel_size, self.hook_channel[size]))
                    self.register_module(f'laplacian{size}_{i}', LaplacianFilter(kernel_size, self.hook_channel[size]))
                self.register_module(f'recover_{size}', ConvBlock(self.hook_channel[size] * (gaussian_group_size * 2 + 1), self.hook_channel[size]))
        
    
    def forward(self, x: Tensor) -> Tensor:
        hook_result = {}
        x = self.conv_in(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for i in range(1, self.layer_count):
            module = self.__getattr__(f'layer{i}')
            x = module(x)
            if module in self.hook_modules.keys():
                if self.gaussian_layer:
                    x = self.pass_gaussian(x, self.hook_modules[module])
                hook_result[self.hook_modules[module]] = x
        return x, hook_result
    
    def pass_gaussian(self, x: Tensor, size: int) -> Tensor:
        '''
        TODO: maybe add a conv layer here for better ulitization of the filters
        '''
        temp = []
        temp.append(x)
        for i in range(self.gaussian_group_size):
            temp.append(self.__getattr__(f'gaussian{size}_{i}')(x))
            temp.append(self.__getattr__(f'laplacian{size}_{i}')(x))
        x = torch.cat(temp, dim=1)
        x = self.__getattr__(f'recover_{size}')(x)
        return x

        


# Decoder module
class Decoder(nn.Module):
    '''
    description: decoder module for image generation
    The input tensor is 1 * 1 * ch_in
    fc_in: linear layer to reshape the input tensor to b_res * b_res * 512
    view: reshape the tensor to 512 * b_res * b_res
    layer1: 
    '''

    def __init__(self,
                 ch_in: int,
                 resolution: int,
                 max_ch: int,
                 base_ch: int,
                 use_res: bool = True,
                 ch_out: int = 3,
                 dropout=0.0
                 ):
        super().__init__()

        # the settings of the img size of hooked layers and base channel number
        if resolution in EMBEDING_RES.keys():
            self.b_res = EMBEDING_RES[resolution]
        else:
            raise Exception(
                "The resolution is not supported yet, please check it.")

        self.hook_size = HOOK_SIZE[resolution]
        
        # initialize the parameters
        self.hook_modules = {}
        self.max_ch = max_ch
        self.base_ch = base_ch
        self.ch_in = ch_in
        
        # initialize the first layer
        cur_res = self.b_res
        cur_ch = self.max_ch

        # TODO: check whether we need to remove fc_in
        # self.fc_in = nn.Linear(ch_in, cur_res * cur_res * cur_ch)
        import math
        self.view_res = int(math.sqrt(ch_in / max_ch))
        # print(self.view_res)
        # self.conv_in = nn.Conv2d(max_ch, max_ch, kernel_size=3, stride=1, padding=1)


        self.layer1 = nn.Conv2d(
            cur_ch, cur_ch, kernel_size=3, stride=1, padding=1)
        self.gn = nn.GroupNorm(32, cur_ch)
        self.silu = nn.ReLU()
        
        # check whether we need to hook the first layer
        if cur_res in self.hook_size:
            self.hook_modules[self.layer1] = cur_res

        # start to build the other layers
        layer_count = 2
        while cur_res < resolution:
            blocks = []
            if cur_ch > self.base_ch:
                b_in = cur_ch
                b_out = cur_ch // 2
            else:
                b_in = cur_ch
                b_out = cur_ch
            if use_res:
                blocks.append(ResBlock(b_in, b_out, dropout))
            else:
                blocks.append(ConvBlock(b_in, b_out, dropout))
            blocks.append(Upsample(b_out))
            # blocks.append(ConvBlock(b_out,b_out,dropout))
            self.register_module(f'layer{layer_count}', nn.Sequential(*blocks))
            cur_res = cur_res * 2
            cur_ch = b_out
            if cur_res in self.hook_size:
                self.hook_modules[self.__getattr__(f'layer{layer_count}')] = cur_res
            layer_count += 1
        self.layer_count = layer_count
        self.conv_out = nn.Conv2d(
            self.base_ch, ch_out, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: Tensor, external:bool= False) -> Tensor:
        hook_result = {}
        # TODO: maybe from here, the image begain
        # x = self.fc_in(x)
        # x = x.view(-1, self.max_ch, self.view_res, self.view_res)
        # x = self.conv_in(x)

        # reshape
        # x = x.view(-1, self.max_ch, self.b_res, self.b_res)
        # pass through the first layer
        x = self.layer1(x)
        x = self.gn(x)
        x = self.silu(x)
        if self.layer1 in self.hook_modules.keys():
            hook_result[self.hook_modules[self.layer1]] = x
        # pass through the other layers
        for i in range(2, self.layer_count):
            module = self.__getattr__(f'layer{i}')
            if isinstance(module, nn.Module):
                x = module(x)
                if module in self.hook_modules.keys():
                    hook_result[self.hook_modules[module]] = x
            else:
                raise Exception("This layer{i} is not a valid torch module.")
        x = self.conv_out(x)
        # TODO: check whether we need to add the last activation function 
        # x = self.silu(x)
        return x, hook_result

    def __repr__(self):
        return f'Decoder with {self.layer_count} layer.'
    
class TRN(nn.Module):
    def __init__(self, model_name: str, resolution: int,
                 gaussian_layer:bool=False,
                 gaussian_group_size:int=3) -> None:
        super().__init__()
        self.encoder = Encoder(model_name, resolution, True, gaussian_layer, gaussian_group_size)
        self.decoder = Decoder(self.encoder.zsize, resolution, self.encoder.max_ch, self.encoder.base_ch)
        self.external = False

    def enable_external(self):
        self.external = True

    def forward(self, x: Tensor) -> Tensor:
        if self.external:
            squeeze = False
            x_device = x.device
            if len(x.shape) < 4:
                    x = x.unsqueeze(0)
                    squeeze = True
            x = x.to(self.encoder.conv_in.weight.device)
            z, encoder_hook = self.encoder(x)
            x_rec, decoder_hook = self.decoder(z)
            if squeeze:
                x, x_rec = x.squeeze(0), x_rec.squeeze(0)
            x, x_rec = x.to(x_device), x_rec.to(x_device)
            return x_rec
        else:
            z, encoder_hook = self.encoder(x)
            x_rec, decoder_hook = self.decoder(z)
            return x_rec, encoder_hook, decoder_hook

def test_decoder():
    test_in = torch.randn([1,1,2048])
    decoder = Decoder([112, 56, 28, 14],2048)
    out, hook_result = decoder(test_in)
    for k, v in hook_result.items():
        print(k, v.shape)
    

def test_basemodel():
    model = BaseModel('resnet50', False, 1000, 32)
    hook_modules = model.hook_detect_fn()
    print(model.hook_channel)
    print(hook_modules)
    # print(model)
    

def test_encoder():
    model = Encoder('resnet18')
    # test_in = torch.randn([1,3,224,224])
    # out = model(test_in)
    # print(out)
    # print(out.keys())
    # print(out[112].shape)
    # print(out[56].shape)
    # print(out[28].shape)
    # print(out[14].shape)

def test_gaussian_filter():
    # test the gaussian filter
    from PIL import Image
    from torchvision import transforms as T
    pic = Image.open('../test244.JPEG')
    to_tensor = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor()
        ]
    )
    to_pil = T.Compose(
        [
            T.ToPILImage()
        ]
    )
    tensor = to_tensor(pic)
    gf = GuassianFilter(5, 3)
    out = gf(tensor.unsqueeze(0))
    out = out.squeeze(0)
    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(to_pil(out))
    plt.figure()
    plt.imshow(to_pil(tensor))


def test_laplacian_filter():
    from PIL import Image
    from torchvision import transforms as T
    pic = Image.open('../test244.JPEG')
    to_tensor = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor()
        ]
    )
    to_pil = T.Compose(
        [
            T.ToPILImage()
        ]
    )
    tensor = to_tensor(pic)
    gf = LaplacianFilter(5, 3)
    out = gf(tensor.unsqueeze(0))
    out = out.squeeze(0)
    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(to_pil(out))
    plt.figure()
    plt.imshow(to_pil(tensor))

def test():
    import sys
    import torch
    torch.autograd.set_detect_anomaly(True)
    model_name = 'resnet18'
    resolution = 32
    sys.path.append('/data/wxl/code')
    # sys.append('/data/wxl/code/FreqDefense')
    from FreqDefense.datasets.datautils import getDataloader
    from torchvision import transforms as T
    from torchvision.utils import save_image, make_grid
    train_loader = getDataloader('imagenet', '/data/wxl/code/FreqDefense/data', 8, 1, True)
    toTensor = T.Compose(
        [T.CenterCrop(resolution)]
    )
    trn = TRN(model_name, resolution)
    teacher = TeacherModel(model_name, resolution)
    trn.train()
    teacher.eval()
    for data in train_loader:
        save_image(data[0], '../results/test/origin.jpg')
        in_test = toTensor(data[0])
        model_hook = teacher(in_test)
        out, hook_result_encoder, hook_result_decoder = trn(in_test)
        save_image(out[0], '../results/test/out.jpg')
        print('test teacher model hook')
        for k,v in model_hook.items():
            print(k, v.shape)
            v = v[0].unsqueeze(1)
            save_image(v, f'../results/test/teacher_{k}.jpg', normalize=True, nrow=8)
        print('test encoder hook')
        for k,v in hook_result_encoder.items():
            print(k, v.shape)
            v = v[0].unsqueeze(1)
            save_image(v, f'../results/test/encoder_{k}.jpg', normalize=True, nrow=8)
        print('test decoder hook')
        for k,v in hook_result_decoder.items():
            print(k, v.shape)
            v = v[0].unsqueeze(1)
            save_image(v, f'../results/test/decoder_{k}.jpg', normalize=True, nrow=8)
        # for size, imgs in out.items():
        #     print(imgs.shape)
        #     img = imgs[0].unsqueeze(1)
        #     save_image(img, f'../results/test/hook_{size}.jpg', normalize=True, nrow=8)
        loss1 = torch.nn.MSELoss()(out, in_test)
        loss2 = torch.tensor(0.0)
        for size, imgs in hook_result_encoder.items():
            loss2 += torch.nn.MSELoss()(imgs, model_hook[size])
            print(f"hook size {size} loss {loss2}")
        loss = loss1 + loss2
        loss.backward()
        for name, param in trn.named_parameters():
            if param.grad is None:
                print(f"Parameter {name} grad is None")
        break


    
if __name__ == '__main__':
    test_laplacian_filter()
    # test()
    # res = 224
    # test_in_size = (1, 3, res, res)
    # model = TeacherModel('resnet18', res)
    # print(summary(model, test_in_size))
    # model2 = Encoder('resnet18', res)
    # model3 = Decoder(model2.zsize, res, model2.max_ch, model2.base_ch)
    # test_in_size2 = (1, model2.max_ch, EMBEDING_RES[res], EMBEDING_RES[res])
    # from torchinfo import summary
    # print(summary(model2, test_in_size))
    # print(summary(model3, test_in_size2))
    # test_decoder()
    # test_gaussian_filter()
    # test_encoder()
    # pass
    