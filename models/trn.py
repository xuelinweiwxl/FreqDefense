'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-04-10 15:41:14
LastEditTime: 2024-04-11 18:48:36
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/models/trn.py
'''

import torch
from torch import nn
from torch.fft import fft2, fftshift, ifft2, ifftshift

import sys
sys.path.append('/data/wxl/code')

from FreqDefense.models.frae import FRAE


class TRN(nn.Module):
    def __init__(self,
                 ch_in=3,
                 bs_chanel=64,
                 resolution=256,
                 ch_muls=[1, 1, 2, 2, 4, 8],
                 dropout=0.0) -> None:
        super().__init__()
        self.mask_gen = FRAE(ch_in, bs_chanel, resolution, ch_muls, dropout)
        self.recover = FRAE(ch_in, bs_chanel, resolution, ch_muls, dropout)
        self.external = False

    def enable_external(self):
        self.external = True

    def forward(self, x):
        if self.external:
            data_device = x.device
            squeeze = False
            if len(x.shape) < 4:
                x = x.unsqueeze(0)
                squeeze = True
            x = x.to(next(self.parameters()).device)

        # get the frequency domain of the x
        x_fft = fft2(x, dim=(-2, -1))
        x_fft = fftshift(x_fft, dim=(-2, -1))
        x_amplitude = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)

        # get frequency mask
        mask = self.mask_gen(x_amplitude)

        # make mask binary
        mask_binary = torch.sign(mask - 0.5).clamp(min=0)

        # apply mask to the amplitude
        x_amplitude_masked = x_amplitude * mask_binary

        # turn to spatial domain
        x_fft_masked = torch.polar(x_amplitude_masked, x_phase)
        x_fft_masked = ifftshift(x_fft_masked, dim=(-2, -1))
        x_masked = ifft2(x_fft_masked, dim=(-2, -1))
        x_masked = torch.real(x_masked)

        # recover the image
        x_rec = self.recover(x_masked)

        if self.external:
            x = x.to(data_device)
            x_rec = x_rec.to(data_device)
            mask = mask.to(data_device)
            if squeeze:
                x_rec = x_rec.squeeze(0)
            return x_rec, mask_binary
        else:
            return x_rec, mask_binary