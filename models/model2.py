'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-04-18 15:21:17
LastEditTime: 2024-04-18 23:21:04
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/models/model2.py
'''

import torch
from torchvision import models
from torch import nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

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
            nn.Conv2d(ch_out, ch_out, Skernel_size=3, stride=1, padding=1),
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
                 num_classes: int = 1000) -> None:
        super().__init__()
        self.model_name = model_name
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained, num_classes=num_classes)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained, num_classes=num_classes)
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained, num_classes=num_classes)
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained, num_classes=num_classes)
        elif model_name == 'vgg19':
            self.model = models.vgg19(pretrained, num_classes=num_classes)
        elif model_name == 'vitb16':
            self.model = models.vit_b_16(pretrained, num_classes=num_classes)
        elif model_name == 'convnext':
            self.model = models.convnext_tiny(
                pretrained, num_classes=num_classes)
        else:
            raise Exception("The model name is not supported.")
        if 'resnet' in model_name:
            self.hook_size = 4
        self.hook_result = {}
        self.hook_id = 0

    def hook_fn(self,
                module: nn.Module,
                input: Tensor,
                output: Tensor):
        self.hook_result[self.hook_id] = output
        self.hook_id += 1

    def get_hook_result(self):
        result = self.hook_result
        self.hook_result = {}
        self.hook_id = 0
        return result

    def forward(self, x: Tensor) -> Tensor:
        if 'resnet' in self.model_name:
            hook1 = self.model.layer1.register_forward_hook(self.hook_fn)
            hook2 = self.model.layer2.register_forward_hook(self.hook_fn)
            hook3 = self.model.layer3.register_forward_hook(self.hook_fn)
            hook4 = self.model.layer4.register_forward_hook(self.hook_fn)
            out = self.model(x)
            hook1.remove()
            hook2.remove()
            hook3.remove()
            hook4.remove()
            return out
        else:
            return self.model(x)

    def __repr__(self):
        return str(self.model)
        return f'model {self.model_name} with {self.model.fc.out_features} classes'

# Decoder module
class Decoder(nn.Module):
    '''
    description: decoder module for image generation
    The input tensor is 1 * 1 * ch_in
    fc_in: 1 * 1 * ch_in ->  7 * 7 * 512
    ch_in: 7 * 7 * 512 -> 14 * 14 * 256 -> 28 * 28 * 128 -> 56 * 56 * 64 -> 112 * 112 * 64 -> 224 * 224 * 64 -> 224 * 224 * ch_out
    TODO: add hook before reaching the final output
    TODO: add four hook for whole model
    TODO: check whether we can add hook in the init function?
    '''


    def __init__(self,
        ch_in: int,
        ch_out: int = 3,
        resolution: int = 224,
        use_res: bool = True,
        dropout = 0.0
    ):
        super().__init__()
        if resolution == 32:
            b_res = 1
        elif resolution == 224:
            b_res = 7
        elif resolution == 256:
            b_res = 8
        else:
            raise Exception("The resolution is not supported yet, please check it.")
        cur_res = b_res
        cur_ch = 512
        base_ch = 64
        self.fc_in = nn.Linear(ch_in, b_res * b_res * cur_ch)
        self.conv_in = nn.Conv2d(cur_ch, cur_ch, kernel_size=3, stride=1, padding=1)
        blocks = []
        while cur_res < resolution:
            if cur_ch > base_ch:
                b_in = cur_ch
                b_out = cur_ch // 2
            else:
                b_in = cur_ch
                b_out = cur_ch
            if use_res:
                blocks.append(ResBlock(b_in, b_out, dropout))
            else:
                blocks.append(ConvBlock(b_in, b_out, dropout))
            blocks.append(Upsample(cur_ch))
            cur_res = cur_res * 2
        self.blocks = nn.Sequential(*blocks)
        self.conv_out = nn.Conv2d(base_ch, ch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        x = self.conv_out(x)
        return x
    
class TRN(nn.Module):
    def __init__(self) -> None:
        super().__init__()


def test():
    import sys
    sys.path.append('/data/wxl/code')
    # sys.append('/data/wxl/code/FreqDefense')
    from FreqDefense.datasets.datautils import getDataloader
    from torchvision import transforms as T
    from torchvision.utils import save_image, make_grid
    train_loader = getDataloader('imagenet', '/data/wxl/code/FreqDefense/data', 8, 1, True)
    toTensor = T.Compose(
        [T.CenterCrop(224)]
    )
    model = BaseModel('resnet18')
    model.eval()
    for data in train_loader:
        save_image(data[0], '../results/test/origin.jpg')
        in_test = toTensor(data[0])
        out = model(in_test)
        for k, v in model.get_hook_result().items():
            print(v.shape)
            for i in range(v.shape[0]):
                vi = v[i].unsqueeze(1)
                save_image(vi, f'../results/test/{k}_{i}.jpg', normalize=True, nrow=8)


    
    

if __name__ == '__main__':
    test()
    
    