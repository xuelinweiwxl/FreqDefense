'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-03-20 10:39:52
LastEditTime: 2024-03-21 15:24:12
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/models/frvae.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
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
# 256 x 256 x 3: [1,2,2,4,8]
# 256 x 256 x 3 -> 256 x 256 x 64 -> 128 x 128 x 64 -> 64 x 64 x 128 -> 32 x 32 x 128 -> 16 x 16 x 256 -> 8 x 8 x 512
# 256 x 256 x 3: [1,1,2,2,4,8]
# 256 x 256 x 3 -> 256 x 256 x 64 -> 128 x 128 x 64 -> 64 x 64 x 64 -> 32 x 32 x 128 -> 16 x 16 x 128 -> 8 x 8 x 256 -> 4 x 4 x 512
class Encoder(nn.Module):
    def __init__(self,
        ch_in = 3,
        chanel = 64,
        resolution = 256,
        ch_muls = [1,1,2,2,4,8],
        z_dim = 1024,
        dropout = 0.0
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(ch_in, chanel, kernel_size=3, stride=1, padding=1)
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
        self.mu = nn.Linear(ch_in * cur_res * cur_res, z_dim)
        self.logvar = nn.Linear(ch_in * cur_res * cur_res, z_dim)
    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.flat(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

# encoder module
# 1024 x 1: [1,2,2,4,8]
# 1024 x 1 -> (4 x 4 x 512) x 1 -> 4 x 4 x 512 -> 8 x 8 x 256 -> 16 x 16 x 128 -> 32 x 32 x 128 -> 64 x 64 x 64 -> 128 x 128 x 64 -> 256 x 256 x 64
class Decoder(nn.Module):
    def __init__(self,
        ch_in = 3,
        chanel = 64,
        resolution = 256,
        ch_muls = [1,1,2,2,4,8],
        z_dim = 1024,
        dropout = 0.0
    ):
        super().__init__()
        cur_res = resolution // (2 ** len(ch_muls))
        self.linear_in = nn.Linear(z_dim, chanel * ch_muls[-1] * cur_res * cur_res)
        self.unflatten = nn.Unflatten(1, (chanel * ch_muls[-1], cur_res, cur_res))
        blocks = []
        ch = chanel * ch_muls[-1]
        for ch_mul in ch_muls[::-1]:
            ch_out = chanel * ch_mul
            blocks.append(ResBlock(ch, ch_out, dropout))
            blocks.append(Upsample(ch_out))
            cur_res = cur_res * 2
            ch = ch_out
        self.blocks = nn.Sequential(*blocks)
        self.conv_out = nn.Conv2d(chanel, ch_in, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = self.linear_in(z)
        x = self.unflatten(x)
        x = self.blocks(x)
        x = self.conv_out(x)
        return x

# vae module
class FRVAE(nn.Module):
    def __init__(self,
        ch_in = 3,
        chanel = 64,
        resolution = 256,
        ch_muls = [1,1,2,2,4,8],
        z_dim = 1024,
        dropout = 0.0
    ):
        super().__init__()
        self.encoder = Encoder(ch_in, chanel, resolution, ch_muls, z_dim, dropout)
        self.decoder = Decoder(ch_in, chanel, resolution, ch_muls, z_dim, dropout)
    
    def reparameterize(self, mu, logvar):
        # log(var^2) = 2 * log(var) -> var = exp(0.5 * log(var^2))
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
            
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

def test():
    # test the encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder().to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    mu, logvar = encoder(x)
    decoder = Decoder().to(device)
    x_hat = decoder(mu)
    print(x_hat.shape)
    print(mu.shape, logvar.shape)
    import time
    from tqdm import tqdm
    tic = time.time()
    batch = 5
    epoch = 10000//batch
    encoder.train()
    decoder.train()
    for i in tqdm(range(epoch)):
        x = torch.randn(batch, 3, 256, 256).to(device)
        mu,log_var = encoder(x)
        x_hat = decoder(mu)
        loss = F.mse_loss(x, x_hat)
        loss.backward()
    toc = time.time()
    print("time: ", toc - tic)
    # writer = SummaryWriter('./runs', comment="comments")
    # writer.add_graph(encoder, (x,))
    # decoder = torch.jit.script(decoder)
    # writer.add_graph(decoder, (mu,))
    # writer.flush()


if __name__ == "__main__":
    test()