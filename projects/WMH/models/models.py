
import torch
from torch import nn
from torch.autograd import Variable

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_sample = nn.Sequential(nn.Conv3d(32, 32, 2, stride=2), 
                                    nn.Tanh())
        
        self.up_sample = nn.Sequential(nn.ConvTranspose3d(48, 32, 2, stride=2), 
                                    nn.ReLU(inplace=True))
        
        self.layers_down = nn.ModuleList([self._conv(2, 16),
                        self._bottle_neck_block(16, 16, 32),
                        self.down_sample, 
                        self._bottle_neck_block(32, 32, 64),
                        self._bottle_neck_block(64, 56, 64),
                        self._bottle_neck_block(64, 56, 86),
                        ])
        
        self.layers_up = nn.ModuleList([self._unconv(86, 64),
                    self._unconv(128, 64),
                    self._unconv(128, 16),
                    self.up_sample,
                    self._unconv(64, 8),
                    nn.ConvTranspose3d(24, 3, 3)
                    ])

    
    def _bottle_neck_block(self, inp_channels, hid_channels, out_channels):
        """ ^_^ """
        return nn.Sequential(
                            torch.nn.Conv3d(inp_channels, hid_channels, 1),
                            nn.BatchNorm3d(hid_channels),
                            nn.ReLU(inplace=True),
                            torch.nn.Conv3d(hid_channels, hid_channels, 3),
                            nn.BatchNorm3d(hid_channels),
                            nn.ReLU(inplace=True),
                            torch.nn.Conv3d(hid_channels, out_channels, 1),
                            nn.BatchNorm3d(out_channels),
                            nn.ReLU(inplace=True)
                            )

    
    def _unconv(self, inp_channels, out_channels):
        return nn.Sequential(nn.ConvTranspose3d(inp_channels, out_channels, 3),
                             nn.Tanh())
    
    
    def _conv(self, inp_channels, out_channels):
        return nn.Sequential(nn.Conv3d(inp_channels, out_channels, 3),
                             nn.BatchNorm3d(out_channels),
                             nn.ReLU(inplace=True))
    
    
    def forward(self, input):
        """-"""
        x0 = self.layers_down[0](input)
        x1 = self.layers_down[1](x0) # [5, x, 26, 26, 16]
        x2 = self.layers_down[2](x1) # [5, x, 13, 13, 8]
        x3 = self.layers_down[3](x2) # [5, x, 11, 11, 6]
        x3_ = nn.Dropout3d(p=0.25)(x3)
        x4 = self.layers_down[4](x3_) # [5, x, 9, 9, 4]
        
        x5 = self.layers_down[5](x4) # [5, x, 7, 7, 2]
        
        x6 = self.layers_up[0](x5) # [5, 16, 9, 9, 4]
        x7 = self.layers_up[1](torch.cat((x4, x6), 1)) # [5, 64, 11, 11, 6]
        x8 = self.layers_up[2](torch.cat((x3, x7), 1)) # [5, 16, 13, 13, 8]
        
        
        x9 = self.layers_up[3](torch.cat((x2, x8), 1)) # [5, 32, 26, 26, 16]
        x9_ = nn.Dropout3d(p=0.1)(x9)
        x10 = self.layers_up[4](torch.cat((x1, x9_), 1)) #[5, 8, 28, 28, 18]
        x11 = self.layers_up[5](torch.cat((x0, x10), 1)) #[5, 8, 28, 28, 18]
        return x11
    