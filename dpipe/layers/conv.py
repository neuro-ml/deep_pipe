from functools import partial

import torch.nn as nn

from .structure import PreActivation, PostActivation


class PreActivationND(PreActivation):
    """
    Performs a sequence of batch_norm, activation, and convolution

        in -> (BN -> activation -> Conv) -> out

    Parameters
    ----------
    in_channels: int
        the number of incoming channels.
    out_channels: int
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
    kwargs
        additional arguments passed to ``layer_module``
    """

    def __init__(self, in_channels, out_channels, *, kernel_size, stride=1, padding=0, bias=True,
                 dilation=1, groups=1, batch_norm_module=None, activation_module=nn.ReLU, conv_module, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, bias=bias, groups=groups, batch_norm_module=batch_norm_module,
                         activation_module=activation_module, layer_module=conv_module, **kwargs)


class PostActivationND(PostActivation):
    """
    Performs a sequence of convolution, batch_norm and activation:

        in -> (Conv -> BN -> activation) -> out

    Parameters
    ----------
    in_channels: int
        the number of incoming channels.
    out_channels: int
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
    kwargs
        additional arguments passed to ``layer_module``
    """

    def __init__(self, in_channels, out_channels, *, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, batch_norm_module=None, activation_module=nn.ReLU, conv_module, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, bias=False, groups=groups, batch_norm_module=batch_norm_module,
                         activation_module=activation_module, layer_module=conv_module, **kwargs)


PreActivation1d = partial(PreActivationND, conv_module=nn.Conv1d, batch_norm_module=nn.BatchNorm1d)
PreActivation2d = partial(PreActivationND, conv_module=nn.Conv2d, batch_norm_module=nn.BatchNorm2d)
PreActivation3d = partial(PreActivationND, conv_module=nn.Conv3d, batch_norm_module=nn.BatchNorm3d)

PostActivation1d = partial(PostActivationND, conv_module=nn.Conv1d, batch_norm_module=nn.BatchNorm1d)
PostActivation2d = partial(PostActivationND, conv_module=nn.Conv2d, batch_norm_module=nn.BatchNorm2d)
PostActivation3d = partial(PostActivationND, conv_module=nn.Conv3d, batch_norm_module=nn.BatchNorm3d)
