from functools import partial

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


class PreActivation(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, *, kernel_size, activation=None, stride=1, padding=0, dilation=1,
                 groups=1, get_activation=None, get_convolution, get_batch_norm):
        super().__init__()
        self.bn = get_batch_norm(num_features=n_chans_in)
        self.activation = infer_activation(activation, get_activation)
        self.conv = get_convolution(in_channels=n_chans_in, out_channels=n_chans_out, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=False, groups=groups)

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

        self.fe = nn.Sequential(PA(n_chans_in, stride=stride), PA(n_chans_out))

        # Shortcut
        spatial_difference = 2 * (kernel_size // 2 - padding)
        self.crop = CenteredCrop(start=[spatial_difference] * dims) if spatial_difference > 0 else identity

        if n_chans_in != n_chans_out:
            self.t = get_convolution(n_chans_in, n_chans_out, kernel_size=1, stride=stride)
        else:
            self.t = identity

        self.shortcut = lambda x: self.t(self.crop(x))

    def forward(self, x):
        return self.fe(x) + self.shortcut(x)


ResBlock2d = partial(ResBlock, get_convolution=nn.Conv2d, get_batch_norm=nn.BatchNorm2d, dims=2)
ResBlock3d = partial(ResBlock, get_convolution=nn.Conv3d, get_batch_norm=nn.BatchNorm3d, dims=3)

PreActivation2d = partial(PreActivation, get_convolution=nn.Conv2d, get_batch_norm=nn.BatchNorm2d)
PreActivation3d = partial(PreActivation, get_convolution=nn.Conv3d, get_batch_norm=nn.BatchNorm3d)

ConvBlock2d: ConvBlock = partial(ConvBlock, get_convolution=nn.Conv2d, get_batch_norm=nn.BatchNorm2d)
ConvBlock3d: ConvBlock = partial(ConvBlock, get_convolution=nn.Conv3d, get_batch_norm=nn.BatchNorm3d)

ConvTransposeBlock2d: ConvBlock = partial(ConvBlock, get_convolution=nn.ConvTranspose2d, get_batch_norm=nn.BatchNorm2d)
ConvTransposeBlock3d: ConvBlock = partial(ConvBlock, get_convolution=nn.ConvTranspose3d, get_batch_norm=nn.BatchNorm3d)
