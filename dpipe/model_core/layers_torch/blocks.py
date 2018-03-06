from functools import partial

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, kernel_size, *, activation=lambda x: x, stride=1,
                 padding=0, dilation=1, convolution, batch_norm, **conv_kwargs):
        super().__init__()
        self.convolution = convolution(in_channels=n_chans_in, out_channels=n_chans_out, kernel_size=kernel_size,
                                       stride=stride, padding=padding, dilation=dilation, bias=False, **conv_kwargs)
        self.batch_norm = batch_norm(num_features=n_chans_out)
        self.activation = activation

    def forward(self, input):
        return self.activation(self.batch_norm(self.convolution(input)))


ConvBlock1d: ConvBlock = partial(ConvBlock, convolution=nn.Conv1d, batch_norm=nn.BatchNorm1d)
ConvBlock2d: ConvBlock = partial(ConvBlock, convolution=nn.Conv2d, batch_norm=nn.BatchNorm2d)
ConvBlock3d: ConvBlock = partial(ConvBlock, convolution=nn.Conv3d, batch_norm=nn.BatchNorm3d)

ConvTransposeBlock1d: ConvBlock = partial(ConvBlock, convolution=nn.ConvTranspose1d, batch_norm=nn.BatchNorm1d)
ConvTransposeBlock2d: ConvBlock = partial(ConvBlock, convolution=nn.ConvTranspose2d, batch_norm=nn.BatchNorm2d)
ConvTransposeBlock3d: ConvBlock = partial(ConvBlock, convolution=nn.ConvTranspose3d, batch_norm=nn.BatchNorm3d)
