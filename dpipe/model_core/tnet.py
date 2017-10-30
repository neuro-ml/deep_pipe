from itertools import zip_longest

import torch
import torch.nn as nn

from dpipe.config import register
from .layers_torch.blocks import ConvBlock2d, ConvBlock3d, ConvTransposeBlock2d, ConvTransposeBlock3d


def _make_tnet(conv_block, conv_transposed_block):
    class TNet(nn.Module):
        def __init__(self, n_chans_in, n_chans_out, activation, structure, stride):
            super().__init__()
            assert all([len(level) == 3 for level in structure[:-1]])
            assert len(structure[-1]) == 1

            down_paths = [level[0] for level in structure[:-1]]
            bridge_paths = [level[1] for level in structure[:-1]] + structure[-1]
            up_paths = [level[2] for level in structure[:-1]]

            down_paths[0] = [n_chans_in, *down_paths[0]]
            bridge_paths = [[*down_level[-1:], *bridge_level]
                            for down_level, bridge_level in zip_longest(down_paths, bridge_paths, fillvalue=[])]

            assert all([bridge_level[-1] < up_level[0] for bridge_level, up_level in zip(bridge_paths, up_paths)])

            kernel_size = 3
            padding = kernel_size // 2

            def build_level(level):
                return nn.Sequential(*[conv_block(n_chans_in=n_chans_in, n_chans_out=n_chans_out, padding=padding,
                                                  kernel_size=kernel_size, activation=activation)
                                       for n_chans_in, n_chans_out in zip(level[:-1], level[1:])])

            self.down_levels = nn.ModuleList([build_level(level) for level in down_paths])
            self.bridge_levels = nn.ModuleList([build_level(level) for level in bridge_paths])
            self.up_levels = nn.ModuleList([build_level(level) for level in up_paths])

            self.down_steps = nn.ModuleList(
                [conv_block(n_chans_in=level[-1], n_chans_out=down_level[0], padding=padding,
                            kernel_size=kernel_size, stride=stride, activation=activation)
                 for level, down_level in zip(down_paths, [*down_paths[1:], bridge_paths[-1]])]
            )

            self.up_steps = nn.ModuleList(
                [conv_transposed_block(n_chans_in=down_level[-1], n_chans_out=level[0] - bridge_level[-1],
                                       padding=padding, kernel_size=kernel_size, stride=stride,
                                       activation=activation)
                 for bridge_level, level, down_level in zip(bridge_paths, up_paths, [*up_paths[1:], bridge_paths[-1]])]
            )

            self.output_layer = conv_block(n_chans_in=up_paths[0][-1], n_chans_out=n_chans_out, kernel_size=1)

        def forward(self, input):
            down_outputs = []
            for level, down_step in zip(self.down_levels, self.down_steps):
                input = level(input)
                down_outputs.append(input)
                input = down_step(input)
            down_outputs.append(input)

            bridge_outputs = [level(input) for input, level in zip(down_outputs, self.bridge_levels)]
            bottom_input = bridge_outputs[-1]
            for bridge_output, up_step, up_level in reversed(list(zip(bridge_outputs[:-1], self.up_steps,
                                                                      self.up_levels))):
                bottom_input = up_level(torch.cat([bridge_output, up_step(bottom_input)], dim=1))
            return self.output_layer(bottom_input)

    return TNet


TNet2d = _make_tnet(ConvBlock2d, ConvTransposeBlock2d)
TNet3d = _make_tnet(ConvBlock3d, ConvTransposeBlock3d)

register('tnet2d')(TNet2d)
register('tnet3d')(TNet3d)
