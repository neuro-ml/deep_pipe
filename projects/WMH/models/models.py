import torch
from torch import nn


class ResBottleNec(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, mode='add'):
        """
        mode: str
            Can take values: 'add', 'concat'.
        """
        super().__init__()
        self.mode = mode
        if mode == 'add':
            out_channels = in_channels
        self.conv_ch_down = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True))
        self.conv_inside = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, 3, bias=False, padding=1),
            nn.ReLU(inplace=True))
        self.conv_up = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, input):
        layer1 = self.conv_ch_down(input)
        layer2 = self.conv_inside(layer1)
        layer3 = self.conv_up(layer2)
        if self.mode == 'add':
            return input + layer3
        else:
            return torch.cat((input, layer3), 1)


class UResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_sample = nn.Sequential(nn.Conv3d(16, 32, 2, stride=2),
                                         nn.Tanh())

        self.up_sample = nn.Sequential(nn.ConvTranspose3d(64, 16, 2, stride=2),
                                       nn.ReLU(inplace=True))

        self.layers_down = nn.ModuleList(
            [self._conv(2, 16), ResBottleNec(16, 16, 16, 'add'),
             self.down_sample, ResBottleNec(32, 32, 16, 'concat'),
             ResBottleNec(48, 48, 48, 'concat'),
             ResBottleNec(96, 64, 32, 'concat')])

        self.layers_up = nn.ModuleList(
            [self._unconv(128, 64), self._unconv(160, 48), self._unconv(96, 32),
             self.up_sample, self._unconv(32, 8), nn.ConvTranspose3d(24, 3, 3)])

    def _unconv(self, inp_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(inp_channels, out_channels, 3, padding=1,
                               bias=False), nn.Tanh())

    def _conv(self, inp_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(inp_channels, out_channels, 3, bias=False),
            nn.ReLU(inplace=True), nn.BatchNorm3d(out_channels))

    def forward(self, input):
        """-"""
        tensors = [input]
        for apply_layer in self.layers_down:
            tensors.append(apply_layer(tensors[-1]))
        counter = len(tensors) - 1
        for apply_layer in self.layers_up:
            if counter == len(tensors) - 1:
                tensors.append(apply_layer(tensors[-1]))
            else:
                tensors.append(
                    apply_layer(torch.cat((tensors[counter], tensors[-1]), 1)))
            counter -= 1
        return tensors[-1]

        # [6, 2, 50, 52, 40]
        # x0 = self.layers_down[0](input)  # [6, 16, 48, 50, 38]
        # x1 = self.layers_down[1](x0)     # [6, 16, 48, 50, 38]
        # x2 = self.layers_down[2](x1)     # [6, 32, 24, 25, 19]
        # x3 = self.layers_down[3](x2)     # [6, 48, 24, 25, 19]
        # x4 = self.layers_down[4](x3)     # [6, 96, 24, 25, 19]
        # x5 = self.layers_down[5](x4)     # [6, 128, 24, 25, 19]
        # x6 = self.layers_up[0](x5)       # [6, 96, 24, 25, 19]
        # x7 = self.layers_up[1](torch.cat((x4, x6), 1))  # [6, 48, 24, 25, 19]
        # x8 = self.layers_up[2](torch.cat((x3, x7), 1))  # [6, 32, 24, 25, 19]
        # x9 = self.layers_up[3](torch.cat((x2, x8), 1))  # [6, 32, 24, 25, 19]
        # x10 = self.layers_up[4](torch.cat((x1, x9), 1))  # [6, 8, 48, 50, 38]
        # x11 = self.layers_up[5](torch.cat((x0, x10), 1)) # [6, 3, 50, 52, 40]
        # return x11


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_sample = nn.Sequential(nn.Conv3d(32, 32, 2, stride=2),
                                         nn.Tanh())

        self.up_sample = nn.Sequential(nn.ConvTranspose3d(48, 32, 2, stride=2),
                                       nn.ReLU(inplace=True))

        self.layers_down = nn.ModuleList(
            [self._conv(2, 16), self._bottle_neck_block(16, 16, 32),
             self.down_sample, self._bottle_neck_block(32, 32, 64),
             self._bottle_neck_block(64, 56, 64), ])

        self.layers_inside = nn.ModuleList([self._bottle_neck_block(64, 56, 86),
                                            # nn.Sequential(
                                            #     nn.ConvTranspose3d(86, 86, 3,
                                            #                        padding=1),
                                            #     nn.BatchNorm3d(86),
                                            #     nn.ReLU(inplace=True)),
                                            self._unconv(86, 64), ])

        self.layers_up = nn.ModuleList(
            [self._unconv(128, 64), self._unconv(128, 16),
             self.up_sample, self._unconv(64, 8), nn.ConvTranspose3d(24, 3, 3)])

    def _bottle_neck_block(self, inp_channels, hid_channels, out_channels):
        """ ^_^ """
        return nn.Sequential(torch.nn.Conv3d(inp_channels, hid_channels, 1),
                             nn.BatchNorm3d(hid_channels),
                             nn.ReLU(inplace=True),
                             torch.nn.Conv3d(hid_channels, hid_channels, 3),
                             nn.BatchNorm3d(hid_channels),
                             nn.ReLU(inplace=True),
                             torch.nn.Conv3d(hid_channels, out_channels, 1),
                             nn.BatchNorm3d(out_channels),
                             nn.ReLU(inplace=True), )

    def _unconv(self, inp_channels, out_channels):
        return nn.Sequential(nn.ConvTranspose3d(inp_channels, out_channels, 3),
                             nn.Tanh())

    def _conv(self, inp_channels, out_channels):
        return nn.Sequential(nn.Conv3d(inp_channels, out_channels, 3),
                             nn.ReLU(inplace=True),
                             nn.BatchNorm3d(out_channels))

    def forward(self, input):
        tensors = [input]
        for apply_layer in self.layers_down:
            tensors.append(apply_layer(tensors[-1]))
        x = tensors[-1]
        for apply_layer in self.layers_inside:
            x = apply_layer(x)
        for apply_layer, left in zip(self.layers_up, reversed(tensors)):
            x = torch.cat((left, x), 1)
            x = apply_layer(x)
        return x

        # x0 = self.layers_down[0](input)
        # x1 = self.layers_down[1](x0)  # [5, x, 26, 26, 16]
        # x2 = self.layers_down[2](x1)  # [5, x, 13, 13, 8]
        # x3 = self.layers_down[3](x2)  # [5, x, 11, 11, 6]
        # x4 = self.layers_down[4](x3)  # [5, x, 9, 9, 4]
        # x5 = self.layers_down[5](x4)  # [5, x, 7, 7, 2]
        # x6 = self.layers_up[0](x5)  # [5, 16, 9, 9, 4]
        # x7 = self.layers_up[1](torch.cat((x4, x6), 1))  # [5, 64, 11, 11, 6]
        # x8 = self.layers_up[2](torch.cat((x3, x7), 1))  # [5, 16, 13, 13, 8]
        # x9 = self.layers_up[3](torch.cat((x2, x8), 1))  # [5, 32, 26, 26, 16]
        # x10 = self.layers_up[4](torch.cat((x1, x9), 1))  # [5, 8, 28, 28, 18]
        # x11 = self.layers_up[5](torch.cat((x0, x10), 1))  # [5, 8, 28, 28, 18]
        # return x11
