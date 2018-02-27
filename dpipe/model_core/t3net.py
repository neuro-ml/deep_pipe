import torch
import torch.nn as nn

from dpipe.model_core.layers_torch.blocks import ConvBlock3d


def get_downsample_op(name, n_chans_in, n_chans_out, kernel_size, activation, stride):
    if name == 'avg':
        return nn.AvgPool3d(kernel_size=3)
    elif name == 'conv':
        return ConvBlock3d(n_chans_in, n_chans_out, kernel_size=kernel_size, activation=activation, stride=stride)
    else:
        raise ValueError(f'Unknown subsampling operation: "{name}"')


def get_upsample_op(name, kernel_size):
    if name == 'neighbour':
        return nn.Upsample(scale_factor=kernel_size)
    else:
        raise ValueError(f"unknown upsample op name: {name}")


def build_level(level, kernel_size, activation):
    return nn.Sequential(*[ConvBlock3d(n_chans_in=n_chans_in, n_chans_out=n_chans_out, kernel_size=kernel_size,
                                       activation=activation)
                           for n_chans_in, n_chans_out in zip(level[:-1], level[1:])])


def get_level_size_decrease(level, kernel_size):
    assert kernel_size % 2 == 1
    return (len(level) - 1) * (kernel_size // 2)


class T3Net(nn.Module):
    def __init__(self, structure, n_chans_out, activation, downsampling, upsampling, stride):
        super().__init__()
        kernel_size = 3

        assert all([len(level) == 3 for level in structure[:-1]])
        assert len(structure[-1]) == 2
        self.structure = structure

        for i, line in enumerate(structure):
            assert len(line[0]) > 0
            line[1] = [line[0][-1], *line[1]]
            if i != len(structure) - 1:
                assert line[1][-1] + structure[i + 1][-1][-1] == line[2][0]

        self.down_levels_ops = nn.ModuleList([build_level(line[0], kernel_size, activation) for line in structure])
        self.bridge_levels_ops = nn.ModuleList([build_level(line[1], kernel_size, activation) for line in structure])
        self.up_levels_ops = nn.ModuleList([build_level(line[2], kernel_size, activation) for line in structure[:-1]])

        self.down_ops = nn.ModuleList([get_downsample_op(downsampling, stride) for _ in range(len(structure) - 1)])
        self.up_ops = nn.ModuleList([get_upsample_op(upsampling, stride) for _ in range(len(structure) - 1)])

        self.bridge_size_decrease = [get_level_size_decrease(line[1], kernel_size) for line in structure[:-1]]

        self.output_layer = ConvBlock3d(n_chans_in=structure[0][2][-1], n_chans_out=n_chans_out, kernel_size=1)

    def forward(self, x):
        down_outputs = []
        for down_level_op, down_step_op in zip(self.down_levels_ops, self.down_ops):
            x = down_level_op(x)
            print(x.size())
            down_outputs.append(x)
            x = down_step_op(x)
        x = self.down_levels_ops[-1](x)
        print(x.size())
        print('Down finished')

        x = self.bridge_levels_ops[-1](x)

        print(x.size())

        for i in reversed(range(len(self.structure) - 1)):
            # Concatenating
            upsampled_output = self.up_ops[i](x)
            print(f'Upsampled: {upsampled_output.size()}')

            slices = [slice(None), slice(None)]
            for us, ds in zip(upsampled_output.size()[2:], down_outputs[i].size()[2:]):
                dif = (ds - us) // 2 - self.bridge_size_decrease[i]
                slices.append(slice(dif, -dif))
            print(f'slices: {slices}')

            bridge_output = self.bridge_levels_ops[i](down_outputs[i][tuple(slices)])
            print(bridge_output.size())
            cat = torch.cat([upsampled_output, bridge_output], dim=1)

            x = self.up_levels_ops[i](cat)
            print(x.size())

        return self.output_layer(x)
