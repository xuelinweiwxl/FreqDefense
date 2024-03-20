"""
* Copyright (c) 2024 Xuelin Wei. All rights reserved.
* SPDX-License-Identifier: MIT
* For full license text, see LICENSE.txt file in the repo root
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# downsample module
class Downsample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

# downsample module2
class Downsample2(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        # left, right, top, bottom
        pad_mode = (0,1,0,1)
        x = F.pad(x, pad_mode, value=0)
        return self.conv(x)

# upsample module
class Upsample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# Resnet module
class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dropout=0.0):
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
            self.shotcut = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        else:
            self.shotcut = nn.Identity()

    def forward(self, x):
        return self.shotcut(x) + self.block(x)


# encoder module
class Encoder(nn.Module):
    def __init__(self,
        ch_in = 3,
        chanel = 64,
        resolution = 256,
        ch_muls = [1,1,2,2,4,8],
        dropout = 0.0
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, chanel, kernel_size=3, stride=1, padding=1)
        ch_in = chanel
        blocks = []
        cur_res = resolution
        for ch_mul in ch_muls:
            ch_out = chanel * ch_mul
            blocks.append(ResBlock(ch_in, ch_out, dropout))
            blocks.append(Downsample(ch_out))
            cur_res = cur_res // 2
            ch_in = ch_out
        self.blocks = nn.Sequential(*blocks)
        self.flat = nn.Flatten()
        self.mu = nn.Linear(ch_in * cur_res * cur_res, ch_in * cur_res * cur_res)
        self.logvar = nn.Linear(ch_in * cur_res * cur_res, ch_in * cur_res * cur_res)
    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.flat(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
        

def main():
    # test the encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder().to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    mu, logvar = encoder(x)
    print(mu.shape, logvar.shape)


if __name__ == "__main__":
    main()