'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-04-10 15:41:14
LastEditTime: 2024-04-10 16:45:32
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/models/trn.py
'''

import torch
from torch import nn
from torch.fft import fft2, fftshift, ifft2, ifftshift
from frae import FRAE


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

    def forward(self, x, external):
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
        mask = (mask > 0.5).float()

        # apply mask to the amplitude
        x_amplitude = x_amplitude * mask

        # turn to spatial domain
        x_fft = torch.polar(x_amplitude, x_phase).real()
        x_fft = ifftshift(x_fft, dim=(-2, -1))
        x = ifft2(x_fft, dim=(-2, -1))

        # recover the image
        x_rec = self.recover(x)

        if self.external:
            x = x.to(data_device)
            x_rec = x_rec.to(data_device)
            mask = mask.to(data_device)
            if squeeze:
                x_rec = x_rec.squeeze(0)
            return x_rec, mask
        else:
            return x_rec, mask