from functools import partial

import numpy as np
import torch.nn as nn

from dpipe.medim.utils import identity
from dpipe.medim.shape_ops import crop_to_shape


class PreActivation(nn.Module):
    """
    Performs a sequence of batch_norm, activation, and convolution

        in -> (BN -> activation -> Conv) -> out

    Parameters
    ----------
    n_chans_in: int
        the number of incoming channels.
    n_chans_out: int
        the number of the ``PreActivation`` output channels.
    kernel_size: int, tuple
        size of the convolving kernel.
    stride: int, tuple, optional
        stride of the convolution. Default is 1.
    padding: int, tuple, optional
        zero-padding added to all spatial sides of the input. Default is 0.
    dilation: int, tuple, optional
        spacing between kernel elements. Default is 1.
    groups: int, optional
        number of blocked connections from input channels to output channels. Default is 1.
    bias: bool
        if ``True``, adds a learnable bias to the output. Default is ``False``
    batch_norm_module: nn.Module
        module to build up batch normalization layer, e.g. ``torch.nn.BatchNorm3d``.
    activation_module: nn.Module
        module to build up activation layer. Default is ``torch.nn.ReLU``.
    conv_module: nn.Module
        module to build up convolution layer with given parameters, e.g. ``torch.nn.Conv3d``.
    """

    def __init__(self, n_chans_in, n_chans_out, *, kernel_size, stride=1, padding=0, bias=True,
                 dilation=1, groups=1, batch_norm_module, activation_module=nn.ReLU, conv_module):
        super().__init__()
        self.bn = batch_norm_module(num_features=n_chans_in)
        self.activation = activation_module()
        self.conv = conv_module(n_chans_in, n_chans_out, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)

    def forward(self, x):
        return self.conv(self.activation(self.bn(x)))


class PostActivation(nn.Module):
    """
    Performs a sequence of convolution, batch_norm and activation:

        in -> (Conv -> BN -> activation) -> out

    Parameters
    ----------
    n_chans_in: int
        the number of incoming channels.
    n_chans_out: int
        the number of the ``PostActivation`` output channels.
    kernel_size: int, tuple
        size of the convolving kernel.
    stride: int, tuple, optional
        stride of the convolution. Default is 1.
    padding: int, tuple, optional
        zero-padding added to all spatial sides of the input. Default is 0.
    dilation: int, tuple, optional
        spacing between kernel elements. Default is 1.
    groups: int, optional
        number of blocked connections from input channels to output channels. Default is 1.
    batch_norm_module: nn.Module
        module to build up batch normalization layer, e.g. ``torch.nn.BatchNorm3d``.
    activation_module: nn.Module
        module to build up activation layer. Default is ``torch.nn.ReLU``.
    conv_module: nn.Module
        module to build up convolution layer with given parameters, e.g. ``torch.nn.Conv3d``.
    """

    def __init__(self, n_chans_in, n_chans_out, *, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 conv_module, batch_norm_module, activation_module=nn.ReLU):
        super().__init__()
        self.conv = conv_module(n_chans_in, n_chans_out, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = batch_norm_module(num_features=n_chans_out)
        self.activation = activation_module()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


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
    n_chans_in: int
        the number of incoming channels.
    n_chans_out: int
        the number of the `ResBlock` output channels.
        Note, if ``n_chans_in`` != ``n_chans_out``, then linear transform will be applied to the shortcut.
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
    """

    def __init__(self, n_chans_in, n_chans_out, *, kernel_size, stride=1, padding=0, dilation=1, bias=False,
                 activation_module=nn.ReLU, conv_module, batch_norm_module):
        super().__init__()
        # ### Features path ###
        pre_activation = partial(
            PreActivation, n_chans_out=n_chans_out, kernel_size=kernel_size, padding=padding, dilation=dilation,
            activation_module=activation_module, conv_module=conv_module, batch_norm_module=batch_norm_module
        )

        self.conv_path = nn.Sequential(pre_activation(n_chans_in, stride=stride, bias=False),
                                       pre_activation(n_chans_out, bias=bias))

        # ### Shortcut ###
        spatial_difference = np.floor(
            np.asarray(dilation) * (np.asarray(kernel_size) - 1) - 2 * np.asarray(padding)
        ).astype(int)
        if not (spatial_difference >= 0).all():
            raise ValueError(f'`spatial_difference` should be greater than zero, {spatial_difference} given')

        if n_chans_in != n_chans_out or stride != 1:
            self.adjust_to_stride = conv_module(n_chans_in, n_chans_out, kernel_size=1, stride=stride, bias=bias)
        else:
            self.adjust_to_stride = identity

    def forward(self, x):
        x_conv = self.conv_path(x)
        x_skip = crop_to_shape(self.adjust_to_stride(x), shape=np.array(x_conv.shape[2:]))
        return x_conv + x_skip


ResBlock3d = partial(ResBlock, conv_module=nn.Conv3d, batch_norm_module=nn.BatchNorm3d)
ResBlock2d = partial(ResBlock, conv_module=nn.Conv2d, batch_norm_module=nn.BatchNorm2d)

PreActivation2d = partial(PreActivation, conv_module=nn.Conv2d, batch_norm_module=nn.BatchNorm2d)
PreActivation3d = partial(PreActivation, conv_module=nn.Conv3d, batch_norm_module=nn.BatchNorm3d)

PostActivation2d = partial(PostActivation, conv_module=nn.Conv2d, batch_norm_module=nn.BatchNorm2d)
PostActivation3d = partial(PostActivation, conv_module=nn.Conv3d, batch_norm_module=nn.BatchNorm3d)
