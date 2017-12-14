import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

from dpipe.config import register


class InitialBlock(nn.Module):
    conv = nn.Conv2d
    pool = nn.MaxPool2d
    batch = nn.BatchNorm2d

    def __init__(self, nchannels, kernel_size=3):
        super().__init__()

        self.convolution = self.conv(nchannels, 16 - nchannels, kernel_size,
                                     stride=2, padding=1, bias=True)
        self.max_pool = self.pool(2, stride=2, ceil_mode=True)
        self.batch_norm = self.batch(16, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.convolution(input), self.max_pool(input)], 1)
        output = self.batch_norm(output)
        return F.relu(output)


class ResBlock(nn.Module):
    conv = nn.Conv2d
    conv_transpose = nn.ConvTranspose2d
    pool = nn.MaxPool2d
    batch = nn.BatchNorm2d
    dropout = nn.Dropout2d

    def __init__(self, input_channels, output_channels, kernel_size=3, downsample=False, upsample=False,
                 dropout_prob=.1, internal_scale=4):
        # it can be either upsampling or downsampling:
        assert not (upsample and downsample)
        super().__init__()

        internal_channels = output_channels // internal_scale
        input_stride = downsample and 2 or 1

        self.downsample = downsample
        self.upsample = upsample
        self.output_channels = output_channels
        self.input_channels = input_channels

        conv_block = [
            self.conv(input_channels, internal_channels, input_stride,
                      stride=input_stride, padding=0, bias=False),
            self.batch(internal_channels, eps=1e-03),
            nn.PReLU(),
        ]

        if upsample:
            conv_block.append(self.conv_transpose(
                internal_channels, internal_channels, kernel_size, stride=2, padding=1, output_padding=1, bias=True
            ))
        else:
            # TODO: use dilated and asymmetric convolutions
            conv_block.append(self.conv(
                internal_channels, internal_channels, kernel_size, stride=1, padding=1, bias=True
            ))

        conv_block.extend([
            self.batch(internal_channels, eps=1e-03),
            nn.PReLU(),
            self.conv(internal_channels, output_channels, 1,
                      stride=1, padding=0, bias=False),
            self.batch(output_channels, eps=1e-03),
            self.dropout(dropout_prob),
        ])

        self.conv_block = nn.Sequential(*conv_block)

        # main path
        if downsample:
            self.max_pool = self.pool(2, stride=2)
        if upsample:
            # TODO: implement unpooling
            self.unpool = self.conv_transpose(
                output_channels, output_channels, kernel_size, stride=2, padding=1, output_padding=1, bias=True
            )

        if output_channels != input_channels:
            self.adjust = nn.Sequential(
                self.conv(input_channels, output_channels, 1,
                          stride=1, padding=0, bias=False),
                self.batch(output_channels, eps=1e-03),
            )

    def forward(self, input):
        conv_path = self.conv_block(input)
        main_path = input

        if self.downsample:
            main_path = self.max_pool(main_path)
        if self.output_channels != self.input_channels:
            main_path = self.adjust(main_path)
        if self.upsample:
            main_path = self.unpool(main_path)

        return F.relu(conv_path + main_path)


class Stage(nn.Module):
    res_block = ResBlock

    def __init__(self, input_channels, output_channels, num_blocks, downsample=False, upsample=False):
        super().__init__()

        blocks = [self.res_block(input_channels, output_channels, downsample=downsample, upsample=upsample)]
        blocks.extend([self.res_block(output_channels, output_channels) for _ in range(num_blocks - 1)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class ENet2D(nn.Module):
    conv_transpose = nn.ConvTranspose2d
    stage = Stage
    initial = InitialBlock

    def __init__(self, n_chans_in, n_chans_out):
        super().__init__()

        self.layers = nn.Sequential(
            self.initial(n_chans_in),
            self.stage(16, 64, 5, downsample=True),
            self.stage(64, 128, 9, downsample=True),
            self.stage(128, 128, 8),
            self.stage(128, 64, 3, upsample=True),
            self.stage(64, 16, 2, upsample=True),
            self.conv_transpose(16, n_chans_out, 1),
        )

    def forward(self, input):
        size = input.size()[2:]
        return torch.nn.functional.upsample_bilinear(self.layers(input), size=size)


class InitialBlock3D(InitialBlock):
    conv = nn.Conv3d
    pool = nn.MaxPool3d
    batch = nn.BatchNorm3d


class ResBlock3D(ResBlock):
    conv = nn.Conv3d
    conv_transpose = nn.ConvTranspose3d
    pool = nn.MaxPool3d
    batch = nn.BatchNorm3d
    dropout = nn.Dropout3d


class Stage3D(Stage):
    res_block = ResBlock3D


class ENet3D(ENet2D):
    conv_transpose = nn.ConvTranspose3d
    stage = Stage3D
    initial = InitialBlock3D
