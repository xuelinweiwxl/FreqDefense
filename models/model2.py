'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-04-18 15:21:17
LastEditTime: 2024-04-22 22:50:25
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/models/model2.py
'''

import torch
from torchvision import models
from torch import nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import functools
from typing import List

'''
TODO: the encoder and teacher model have same basic structure,
    there are still some differences between them
    1. The encoder don't have the last layers of the basic model
    2. We need to add some gaussian group filters to the encoder
    3. We need to compare the outputs of middle layers of the encoder and teacher model
    So we can create a father class for this two models to reduce the duplicate code.
'''

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
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, ch_out),
            nn.SiLU(),
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
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, ch_out),
            nn.SiLU(),
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
        if 'resnet' in model_name:
            self.hook_size = 4
        else:
            raise Exception("The model name is not supported.")

        self.hook_result = {}
        self.model_name = model_name
        self.pretrained = pretrained

        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained, num_classes=num_classes)
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
        
        # first: determine the size we want to hook and compare
        if input_resolution == 224:
            self.hook_size = [56, 28, 14, 7]
        elif input_resolution == 256:
            self.hook_size = [64, 32, 16, 8]
        elif input_resolution == 32:
            self.hook_size = [8, 4, 2]
        else:
            raise Exception("The input resolution is not supported.")

    def hook_fn(self,
                module: nn.Module,
                input: Tensor,
                output: Tensor,
                size: int) -> None:
        self.hook_result[size] = output
    
    # Detect where to place a hook according to the model type
    def hook_detect_fn(self) -> None:
        # hook handles
        hook_modules = {}
        
        # depending on the model type, we need to find the module to hook
        # check if the model is pretrained, if the model is pretrained, we need to find the correct module to hook
        if self.pretrained:
            if 'resnet' in self.model_name:
                hook_modules[self.hook_size[0]] = self.model.layer1 # 56 64 8
                hook_modules[self.hook_size[1]] = self.model.layer2 # 28 32 4
                hook_modules[self.hook_size[2]] = self.model.layer3 # 14 16 2
                hook_modules[self.hook_size[3]] = self.model.layer4 # 7 8 1
            else:
                raise Exception("The model name is not supported.")
        
        return hook_modules
    
    def forward(self, x: Tensor) -> None:
        pass

    def get_hook_result(self):
        result = self.hook_result
        self.hook_result = {}
        return result

    def __repr__(self):
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
        hook_modules = self.hook_detect_fn()
        for size, module in hook_modules.items():
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
        self.gaussian_filter = nn.Conv2d(channel, channel, kernel_size, padding=kernel_size//2, bias=False)
        self.sigma = nn.Parameter(torch.randn(1))
        self.channel = channel
    
    def get_gaussian_kernel(self) -> Tensor:
        n = self.kernel_size // 2
        kernel = torch.exp(-(torch.arange(-n, n + 1) ** 2) / (2 * self.sigma ** 2))
        kernel = kernel.unsqueeze(0)
        kernel = kernel.mul(kernel.t())
        kernel = kernel / kernel.sum().sum()
        temp_weight = torch.zeros_like(self.gaussian_filter.weight.data)
        index = range(self.channel)
        temp_weight[index, index, :, :] = kernel
        self.gaussian_filter.weight.data.copy_(temp_weight)

    def forward(self, x: Tensor) -> Tensor:
        self.get_gaussian_kernel()
        x = self.gaussian_filter(x)
        return x

# Encoder module
class Encoder(BaseModel):
    '''
    description: using the same structure as the teacher model
    feature:
    1. add gaussian group filters between layers 
    TODO: have to check whether the hook can save gradient during backpropagation
    '''

    def __init__(self,
                  model_name: str,
                    input_resolution: int = 224,
                    gaussian_group_size: int = 5) -> None:
        super().__init__(model_name, False, 1000, input_resolution)

        self.gaussian_group_size = gaussian_group_size

        if 'resnet' in model_name:
            # get the layers from the base model
            self.conv1 = self.model.conv1
            self.bn1 = self.model.bn1
            self.relu = self.model.relu
            self.maxpool = self.model.maxpool
            self.layer1 = self.model.layer1
            self.layer2 = self.model.layer2
            self.layer3 = self.model.layer3
            self.layer4 = self.model.layer4

            for i in range(gaussian_group_size):
                self.register_module(f'gaussian{self.hook_size[0]}{i}', GuassianFilter(5, self.conv1.out_channels))
                self.register_module(f'gaussian{self.hook_size[1]}{i}', GuassianFilter(5, self.layer1[0].conv1.out_channels))
                self.register_module(f'gaussian{self.hook_size[2]}{i}', GuassianFilter(5, self.layer2[0].conv1.out_channels))
                self.register_module(f'gaussian{self.hook_size[3]}{i}', GuassianFilter(5, self.layer3[0].conv1.out_channels))

        else:
            raise Exception("The model name is not supported.")
        
    
    def pass_gaussian(self, x: Tensor, size: int) -> Tensor:
        result = x
        '''
        TODO: maybe add a conv layer here for better ulitization of the filters
        '''
        for i in range(self.gaussian_group_size):
            result += self.__getattr__(f'gaussian{size}{i}')(x)
        return result

    def forward(self, x: Tensor) -> Tensor:
        hook_result = {}
        if 'resnet' in self.model_name:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.pass_gaussian(x, self.hook_size[0])
            hook_result[self.hook_size[0]] = x
            x = self.layer2(x)
            x = self.pass_gaussian(x, self.hook_size[1])
            hook_result[self.hook_size[1]] = x
            x = self.layer3(x)
            x = self.pass_gaussian(x, self.hook_size[2])
            hook_result[self.hook_size[2]] = x
            x = self.layer4(x)
            x = self.pass_gaussian(x, self.hook_size[3])
            hook_result[self.hook_size[3]] = x
            x = torch.flatten(x, 1)
        else:
            raise Exception("The model name is not supported.")
        return x, hook_result 

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
                 hook_size: List[int],
                 ch_in: int,
                 resolution: int = 224,
                 use_res: bool = True,
                 ch_out: int = 3,
                 dropout=0.0
                 ):
        super().__init__()
        if resolution == 32:
            self.b_res = 1
        elif resolution == 224:
            self.b_res = 7
        elif resolution == 256:
            self.b_res = 8
        else:
            raise Exception(
                "The resolution is not supported yet, please check it.")
        self.hook_modules = {}
        self.max_ch = 512
        self.base_ch = 64
        cur_res = self.b_res
        cur_ch = self.max_ch
        self.fc_in = nn.Linear(ch_in, cur_res * cur_res * cur_ch)
        self.layer1 = nn.Conv2d(
            cur_ch, cur_ch, kernel_size=3, stride=1, padding=1)
        if cur_res in hook_size:
            self.hook_modules[self.layer1] = cur_res
        # TODO: add group normalization and SiLU activation
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
            blocks.append(ConvBlock(b_out,b_out,dropout))
            self.register_module(f'layer{layer_count}', nn.Sequential(*blocks))
            cur_res = cur_res * 2
            cur_ch = b_out
            if cur_res in hook_size:
                self.hook_modules[self.__getattr__(f'layer{layer_count}')] = cur_res
            layer_count += 1
        self.layer_count = layer_count
        self.conv_out = nn.Conv2d(
            self.base_ch, ch_out, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: Tensor) -> Tensor:
        hook_result = {}
        x = self.fc_in(x)
        # reshape
        x = x.view(-1, self.max_ch, self.b_res, self.b_res)
        for i in range(1, self.layer_count):
            module = self.__getattr__(f'layer{i}')
            if isinstance(module, nn.Module):
                x = module(x)
                if module in self.hook_modules.keys():
                    hook_result[self.hook_modules[module]] = x
            else:
                raise Exception("This layer{i} is not a valid torch module.")
        x = self.conv_out(x)
        return x, hook_result

    def __repr__(self):
        return f'Decoder with {self.layer_count} layer.'

def test_decoder():
    test_in = torch.randn([1,1,2048])
    decoder = Decoder([112, 56, 28, 14],2048)
    out, hook_result = decoder(test_in)
    for k, v in hook_result.items():
        print(k, v.shape)
    

# model = models.resnet18(pretrained=True)
# print(model)
# test_in = torch.randn([1,3,32,32])
# print(model.children())
# for i, layer in enumerate(model.features):
#     test_in = layer(test_in)
#     print(i, test_in.shape)
# print(out.shape)
class TRN(nn.Module):
    def ___init__(self,) -> None:
        super().__init__()
        self.basemodel = BaseModel('resnet18')
        self.decoder = Decoder()
        self.encoder = Encoder()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.basemodel(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

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


def test():
    import sys
    model_name = 'resnet50'
    sys.path.append('/data/wxl/code')
    # sys.append('/data/wxl/code/FreqDefense')
    from FreqDefense.datasets.datautils import getDataloader
    from torchvision import transforms as T
    from torchvision.utils import save_image, make_grid
    train_loader = getDataloader('imagenet', '/data/wxl/code/FreqDefense/data', 8, 1, True)
    toTensor = T.Compose(
        [T.CenterCrop(256)]
    )
    model = TeacherModel(model_name, 256)
    encoder = Encoder(model_name, 256)
    decoder = Decoder([128, 64, 32, 16], 512 * 8 * 8, 256)
    # decoder = Decoder([112, 56, 28, 14], 512 * 7 * 7)
    model.eval()
    encoder.eval()
    decoder.eval()
    for data in train_loader:
        save_image(data[0], '../results/test/origin.jpg')
        in_test = toTensor(data[0])
        out = model(in_test)
        z, hook_result_encoder = encoder(in_test)
        print(f'zshape: {z.shape}')
        x_rec, hook_result_decoder = decoder(z)
        print(f'x_rec: {x_rec.shape}')
        print('test teacher model hook')
        for k,v in out.items():
            print(k, v.shape)
            v = v[0].unsqueeze(1)
            save_image(v, f'../results/test/teacher_{k}.jpg', normalize=True, nrow=8)
        print('test encoder hook')
        for k,v in hook_result_encoder.items():
            print(k, v.shape)
            v = v[0].unsqueeze(1)
            save_image(v, f'../results/test/encoder_{k}.jpg', normalize=True, nrow=8)
        print('test decoder hook')
        print(hook_result_decoder.keys())
        for k,v in hook_result_decoder.items():
            print(k, v.shape)
            v = v[0].unsqueeze(1)
            save_image(v, f'../results/test/decoder_{k}.jpg', normalize=True, nrow=8)
        # for size, imgs in out.items():
        #     print(imgs.shape)
        #     img = imgs[0].unsqueeze(1)
        #     save_image(img, f'../results/test/hook_{size}.jpg', normalize=True, nrow=8)
        break


    
if __name__ == '__main__':
    test()
    # test_decoder()
    # test_gaussian_filter()
    # test_encoder()
    # pass
    