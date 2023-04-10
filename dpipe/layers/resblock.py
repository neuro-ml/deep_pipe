from functools import partial

import numpy as np
import torch.nn as nn

from dpipe.layers import PreActivationND
from dpipe.im.utils import identity
from dpipe.im.shape_ops import crop_to_shape


class ResBlock(nn.Module):
    """
    Performs a sequence of two convolutions with residual connection (Residual Block).

    ..
        in ---> (BN --> activation --> Conv) --> (BN --> activation --> Conv) -- + --> out
            |                                                                    ^
            |                                                                    |
             --------------------------------------------------------------------

    Parameters
    ----------
    in_channels: int
        the number of incoming channels.
    out_channels: int
        the number of the `ResBlock` output channels.
        Note, if ``in_channels`` != ``out_channels``, then linear transform will be applied to the shortcut.
    kernel_size: int, tuple
        size of the convolving kernel.
    stride: int, tuple, optional
        stride of the convolution. Default is 1.
        Note, if stride is greater than 1, then linear transform will be applied to the shortcut.
    padding: int, tuple, optional
        zero-padding added to all spatial sides of the input. Default is 0.
    dilation: int, tuple, optional
        spacing between kernel elements. Default is 1.
    bias: bool
        if ``True``, adds a learnable bias to the output. Default is ``False``.
    activation_module: None, nn.Module, optional
        module to build up activation layer.  Default is ``torch.nn.ReLU``.
    conv_module: nn.Module
        module to build up convolution layer with given parameters, e.g. ``torch.nn.Conv3d``.
    batch_norm_module: nn.Module
        module to build up batch normalization layer, e.g. ``torch.nn.BatchNorm3d``.
    kwargs
        additional arguments passed to ``conv_module``.
    """

    def __init__(self, in_channels, out_channels, *, kernel_size, stride=1, padding=0, dilation=1, bias=False,
                 activation_module=nn.ReLU, conv_module, batch_norm_module, **kwargs):
        super().__init__()
        # ### Features path ###
        pre_activation = partial(
            PreActivationND, kernel_size=kernel_size, padding=padding, dilation=dilation,
            activation_module=activation_module, conv_module=conv_module, batch_norm_module=batch_norm_module, **kwargs
        )

        self.conv_path = nn.Sequential(pre_activation(in_channels, out_channels, stride=stride, bias=False),
                                       pre_activation(out_channels, out_channels, bias=bias))

        # ### Shortcut ###
        spatial_difference = np.floor(
            np.asarray(dilation) * (np.asarray(kernel_size) - 1) - 2 * np.asarray(padding)
        ).astype(int)
        if not (spatial_difference >= 0).all():
            raise ValueError(f"The output's shape cannot be greater than the input's shape. ({spatial_difference})")

        if in_channels != out_channels or stride != 1:
            self.adjust_to_stride = conv_module(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
        else:
            self.adjust_to_stride = identity

    def forward(self, x):
        x_conv = self.conv_path(x)
        shape = x_conv.shape[2:]
        axes = range(-len(shape), 0)
        x_skip = crop_to_shape(self.adjust_to_stride(x), shape=shape, axis=axes)
        return x_conv + x_skip


ResBlock1d = partial(ResBlock, conv_module=nn.Conv1d, batch_norm_module=nn.BatchNorm1d)
ResBlock2d = partial(ResBlock, conv_module=nn.Conv2d, batch_norm_module=nn.BatchNorm2d)
ResBlock3d = partial(ResBlock, conv_module=nn.Conv3d, batch_norm_module=nn.BatchNorm3d)
