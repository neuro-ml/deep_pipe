from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional

from .layers_torch.blocks import ConvBlock3d, ConvTransposeBlock3d

context_slice = tuple(2 * [slice(None)] + 3 * [slice(1, -1, 3)])

downsample_ops = {
    'avg': partial(nn.functional.avg_pool3d, kernel_size=3),
    'max': partial(nn.functional.max_pool3d, kernel_size=3),
    # 'sample': lambda x: x[context_slice],
}


class DeepMedicEls(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, downsample='avg', upsample='neighbour',
                 activation=nn.ReLU(inplace=True), dropout=False):
        super().__init__()

        def build_path(path_structure, kernel_size, dropout=False):
            path = []

            if dropout:
                path.append(nn.Dropout3d())

            for n_chans_in, n_chans_out in zip(path_structure[:-1], path_structure[1:]):
                path.append(ConvBlock3d(n_chans_in, n_chans_out, kernel_size, activation=activation))

                if dropout:
                    path.append(nn.Dropout3d())
            return nn.Sequential(*path)

        path_structure = [n_chans_in, 30, 30, 40, 40, 40, 40, 50, 50]
        self.detailed_path = build_path(path_structure, 3)
        self.context_path = build_path(path_structure, 3)

        self.downsample_op = downsample_ops[downsample]
        self.upsample_op = partial(nn.functional.upsample, scale_factor=3)

        common_path_structure = [2 * path_structure[-1], 150, 150]
        self.common_path = build_path(common_path_structure, 1, dropout=True)

        self.logits_layer = ConvBlock3d(common_path_structure[-1], n_chans_out, 1)

    def forward(self, detailed_input, context_input):
        detailed = self.detailed_path(detailed_input)
        context = self.upsample_op(self.context_path(self.downsample_op(context_input)))

        common_input = torch.cat([detailed, context], 1)
        common = self.common_path(common_input)

        return self.logits_layer(common)
