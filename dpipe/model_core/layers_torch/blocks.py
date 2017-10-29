import torch.nn as nn


def _make_conv_block(convolution, batch_norm):
    class ConvBlock(nn.Module):
        def __init__(self, n_chans_in, n_chans_out, kernel_size, activation=lambda x: x, stride=1,
                     padding=0, dilation=1):
            super().__init__()
            self.convolution = convolution(in_channels=n_chans_in, out_channels=n_chans_out, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=False)
            self.batch_norm = batch_norm(num_features=n_chans_out)
            self.activation = activation

        def forward(self, input):
            return self.activation(self.batch_norm(self.convolution(input)))

    return ConvBlock


ConvBlock1d = _make_conv_block(nn.Conv1d, nn.BatchNorm1d)
ConvBlock2d = _make_conv_block(nn.Conv2d, nn.BatchNorm2d)
ConvBlock3d = _make_conv_block(nn.Conv3d, nn.BatchNorm3d)

ConvTransposeBlock1d = _make_conv_block(nn.ConvTranspose1d, nn.BatchNorm1d)
ConvTransposeBlock2d = _make_conv_block(nn.ConvTranspose2d, nn.BatchNorm2d)
ConvTransposeBlock3d = _make_conv_block(nn.ConvTranspose3d, nn.BatchNorm3d)
