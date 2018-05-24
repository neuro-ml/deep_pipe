from functools import partial

import numpy as np
import torch.nn as nn
from .layer import CenteredCrop


def identity(x):
    return x


def infer_activation(activation, get_activation):
    assert (activation is None) ^ (get_activation is None), 'Have to provide either activation or get_activation.'
    return activation if get_activation is None else get_activation()


class ConvBlock(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, *, kernel_size, activation=None, stride=1, padding=0, dilation=1,
                 groups=1, get_activation=None, get_convolution, get_batch_norm):
        super().__init__()
        self.conv = get_convolution(in_channels=n_chans_in, out_channels=n_chans_out, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = get_batch_norm(num_features=n_chans_out)
        self.activation = infer_activation(activation, get_activation)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


def make_res_init(structure, kernel_size, activation, padding=0):
    if len(structure) == 2:
        return nn.Sequential(nn.Conv3d(structure[0], structure[1], kernel_size=kernel_size, padding=padding))
    else:
        return nn.Sequential(ConvBlock3d(structure[0], structure[1], kernel_size=kernel_size, padding=padding,
                                         activation=activation),
                             *make_res_init(structure[1:], kernel_size=kernel_size, padding=padding,
                                            activation=activation))


class PreActivation(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, *, kernel_size, activation=None, stride=1, padding=0, bias=True,
                 dilation=1, groups=1, get_activation=None, get_convolution, get_batch_norm):
        super().__init__()
        self.bn = get_batch_norm(num_features=n_chans_in)
        self.activation = infer_activation(activation, get_activation)
        self.conv = get_convolution(in_channels=n_chans_in, out_channels=n_chans_out, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)

    def forward(self, x):
        return self.conv(self.activation(self.bn(x)))


class ResBlock(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, *, kernel_size=3, activation=None, stride=1, padding=0, dilation=1,
                 get_activation=None, get_convolution, get_batch_norm, dims):
        super().__init__()

        # Features
        PA = partial(PreActivation, n_chans_out=n_chans_out, kernel_size=kernel_size, activation=activation,
                     padding=padding, dilation=dilation, get_activation=get_activation, get_convolution=get_convolution,
                     get_batch_norm=get_batch_norm)

        self.conv_path = nn.Sequential(PA(n_chans_in, stride=stride, bias=False), PA(n_chans_out))

        # Shortcut
        spatial_difference = np.broadcast_to(2 * (kernel_size // 2 - padding), dims)
        assert (spatial_difference >= 0).all()
        self.crop = CenteredCrop(start=spatial_difference) if (spatial_difference > 0).any() else identity

        if n_chans_in != n_chans_out or stride != 1:
            self.adjust_to_stride = get_convolution(n_chans_in, n_chans_out, kernel_size=1, stride=stride)
        else:
            self.adjust_to_stride = identity

    def forward(self, x):
        return self.conv_path(x) + self.adjust_to_stride(self.crop(x))


ResBlock2d = partial(ResBlock, get_convolution=nn.Conv2d, get_batch_norm=nn.BatchNorm2d, dims=2)
ResBlock3d = partial(ResBlock, get_convolution=nn.Conv3d, get_batch_norm=nn.BatchNorm3d, dims=3)

PreActivation2d = partial(PreActivation, get_convolution=nn.Conv2d, get_batch_norm=nn.BatchNorm2d)
PreActivation3d = partial(PreActivation, get_convolution=nn.Conv3d, get_batch_norm=nn.BatchNorm3d)

ConvBlock2d: ConvBlock = partial(ConvBlock, get_convolution=nn.Conv2d, get_batch_norm=nn.BatchNorm2d)
ConvBlock3d: ConvBlock = partial(ConvBlock, get_convolution=nn.Conv3d, get_batch_norm=nn.BatchNorm3d)

ConvTransposeBlock2d: ConvBlock = partial(ConvBlock, get_convolution=nn.ConvTranspose2d, get_batch_norm=nn.BatchNorm2d)
ConvTransposeBlock3d: ConvBlock = partial(ConvBlock, get_convolution=nn.ConvTranspose3d, get_batch_norm=nn.BatchNorm3d)
