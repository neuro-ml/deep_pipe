import torch
import torch.nn as nn


def compose_blocks(structure, get_block):
    return nn.Sequential(*[
        get_block(n_chans_in, n_chans_out) for n_chans_in, n_chans_out in zip(structure[:-1], structure[1:])
    ])


class SplitCat(nn.Module):
    def __init__(self, *paths):
        super().__init__()
        self.paths = nn.ModuleList(list(paths))

    def forward(self, x):
        return torch.cat([path(x) for path in self.paths], dim=1)
