from functools import partial

import torch
import torch.nn as nn

from dpipe.model_core.layers_torch.blocks import ConvBlock3d


def get_downsample_op(name, n_chans_in, n_chans_out, kernel_size, activation, stride):
    if name == 'avg':
        return nn.AvgPool3d(kernel_size=stride)
    elif name == 'max':
        return nn.MaxPool3d(kernel_size=stride)
    elif name == 'conv':
        return ConvBlock3d(n_chans_in, n_chans_out, kernel_size=kernel_size, activation=activation, stride=stride)
    else:
        raise ValueError(f'Unknown subsampling operation: "{name}"')


def get_upsample_op(name, stride):
    if name == 'neighbour':
        return nn.Upsample(scale_factor=stride)
    else:
        raise ValueError(f"unknown upsample op name: {name}")


def build_level(level, kernel_size, activation, padding):
    return nn.Sequential(*[ConvBlock3d(n_chans_in=n_chans_in, n_chans_out=n_chans_out, kernel_size=kernel_size,
                                       activation=activation, padding=padding)
                           for n_chans_in, n_chans_out in zip(level[:-1], level[1:])])


def get_level_size_decrease(level, kernel_size, padding):
    assert kernel_size % 2 == 1
    return (len(level) - 1) * (kernel_size // 2 - padding)


class T3Net(nn.Module):
    def __init__(self, structure, stride, kernel_size=3, downsampling='avg', upsampling='neighbour',
                 activation=nn.ReLU(inplace=True), padding=0):
        super().__init__()

        assert all([len(level) == 3 for level in structure[:-1]])
        assert len(structure[-1]) == 2

        for i, line in enumerate(structure):
            assert len(line[0]) > 0
            line[1] = [line[0][-1], *line[1]]
            if i != len(structure) - 1:
                assert line[1][-1] + structure[i + 1][-1][-1] == line[2][0]
        self.structure = structure

        bl = partial(build_level, kernel_size=kernel_size, padding=padding, activation=activation)

        self.down_levels_ops = nn.ModuleList([bl(line[0]) for line in structure])
        self.bridge_levels_ops = nn.ModuleList([bl(line[1]) for line in structure])
        self.up_levels_ops = nn.ModuleList([bl(line[2]) for line in structure[:-1]])

        self.down_ops = nn.ModuleList([
            get_downsample_op(downsampling, structure[i][0][-1], structure[i+1][0][0], kernel_size=kernel_size,
                              activation=activation, stride=stride)
            for i in range(len(structure) - 1)
        ])
        self.up_ops = nn.ModuleList([get_upsample_op(upsampling, stride=stride) for _ in range(len(structure) - 1)])

        self.bridge_size_decrease = [get_level_size_decrease(line[1], kernel_size=kernel_size, padding=padding)
                                     for line in structure[:-1]]

    def forward(self, x):
        down_outputs = []
        for down_level_op, down_step_op in zip(self.down_levels_ops, self.down_ops):
            x = down_level_op(x)
            print(f'Down path output: {x.size()}')
            down_outputs.append(x)
            x = down_step_op(x)
            print(f'Downsampling output: {x.size()}')
        x = self.down_levels_ops[-1](x)
        print(f'Down path output {x.size()}')
        print('Down finished')

        x = self.bridge_levels_ops[-1](x)

        for i in reversed(range(len(self.structure) - 1)):
            # Concatenating
            print(f'Bridge output: {x.size()}')
            upsampled_output = self.up_ops[i](x)
            print(f'Upsampled: {upsampled_output.size()}')

            slices = [slice(None), slice(None)]
            for us, ds in zip(upsampled_output.size()[2:], down_outputs[i].size()[2:]):
                dif = (ds - us) // 2 - self.bridge_size_decrease[i]
                slices.append(slice(dif, -dif if dif != 0 else None))
            #print(f'slices: {slices}')

            sliced = down_outputs[i][tuple(slices)]
            print(f'Bridge input: {sliced.size()}')
            bridge_output = self.bridge_levels_ops[i](sliced )
            print(f'Bridge output: {bridge_output.size()}')
            cat = torch.cat([upsampled_output, bridge_output], dim=1)

            x = self.up_levels_ops[i](cat)
            print(f'Up path output: {x.size()}')

        return x