import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def cb(in_chans, out_chans, kernel_size, relu=False):
    model = [nn.Conv3d(in_chans, out_chans, kernel_size, bias=False),
             nn.BatchNorm3d(out_chans)]
    if relu:
        model.append(nn.ReLU(inplace=True))

    return nn.Sequential(*model)


class ResBlock(nn.Module):
    def __init__(self, in_chans, int_chans, out_chans, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_chans = in_chans
        self.int_chans = int_chans
        self.out_chans = out_chans

        self.fe = nn.Sequential(
            cb(in_chans, int_chans, 1, relu=True),
            cb(int_chans, int_chans, kernel_size, relu=True),
            cb(int_chans, out_chans, 1),
        )

        if in_chans != out_chans:
            self.transformer = cb(in_chans, out_chans, 1)

    def forward(self, input):
        out = self.fe(input)

        s = self.kernel_size // 2
        if self.in_chans != self.out_chans:
            out += self.transformer(input[:, :, s:-s, s:-s, s:-s])
        else:
            out += input[:, :, s:-s, s:-s, s:-s]

        return F.relu(out, inplace=True)


class Model(torch.nn.Module):
    def __init__(self, blocks, n_classes, kernel_size):
        super().__init__()

        fe = []
        for n_chans_prev, n_chans in zip(blocks, blocks[1:]):
            fe.append(
                ResBlock(n_chans_prev, n_chans // 4, n_chans, kernel_size))

        fe.extend([cb(n_chans, n_classes, 1, relu=False)])

        self.model = nn.Sequential(*fe)

    def forward(self, input):
        return self.model(input)
