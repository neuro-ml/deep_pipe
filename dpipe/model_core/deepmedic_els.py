import torch
import torch.nn as nn

from .layers_torch.blocks import ConvBlock3d


class DeepMedicEls(nn.Module):
    def __init__(self, n_chans_in, n_chans_out):
        super().__init__()
        activation = nn.functional.relu

        def build_path(path_structure, kernel_size):
            path = []
            for n_chans_in, n_chans_out in zip(path_structure[:-1], path_structure[1:]):
                path.append(ConvBlock3d(n_chans_in, n_chans_out, kernel_size, activation=activation))
            return nn.Sequential(*path)

        path_structure = [n_chans_in, 30, 30, 40, 40, 40, 40, 50, 50]
        self.detailed_path = build_path(path_structure, 3)
        self.context_path = build_path(path_structure, 3)

        self.context_slice = tuple(2 * [slice(None)] + 3 * [slice(1, -1, 3)])

        common_path_structure = [2 * path_structure[-1], 150, 150]
        self.common_path = build_path(common_path_structure, 1)

        self.logits_layer = ConvBlock3d(common_path_structure[-1], n_chans_out, 1)

    def forward(self, detailed_input, context_input):
        detailed = self.detailed_path(detailed_input)
        context = nn.functional.upsample(self.context_path(context_input[self.context_slice]), scale_factor=3)

        common_input = torch.cat([detailed, context], 1)
        common = self.common_path(common_input)

        return self.logits_layer(common)
