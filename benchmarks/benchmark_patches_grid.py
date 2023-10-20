import numpy as np
import torch
from torch import nn

from dpipe.predict import patches_grid
from dpipe.torch.model import inference_step


class IdentityWithParams(nn.Conv3d):
    """Needed for determining device"""
    def forward(self, x):
        return x


class TimeSuite:
    params = (('float16', 'float32', 'float64'),)
    param_names = ('dtype',)
    timeout = 300

    def setup(self, dtype):
        self.inp = np.random.randn(512, 512, 512).astype(dtype)
        self.predict = patches_grid(200, 100, axis=-1)(lambda x: x)

    def time_patches_grid(self, dtype):
        self.predict(self.inp)

    def peakmem_patches_grid(self, dtype):
        self.predict(self.inp)


class TimeTorchSuite:
    params = ((False, True),)
    param_names = ('amp',)

    def setup(self, amp):
        self.inp = np.random.randn(512, 512, 512).astype('float32')

        try:
            inference_step(self.inp[None, None], architecture=IdentityWithParams(1, 1, kernel_size=3), in_dtype=torch.float32)
            self.predict = patches_grid(200, 100, axis=-1)(
                lambda x: inference_step(
                    x[None, None],
                    architecture=IdentityWithParams(1, 1, kernel_size=3),
                    amp=amp, out_dtype=torch.float32,
                )[0][0]
            )
        except TypeError:
            self.predict = patches_grid(200, 100, axis=-1)(
                lambda x: inference_step(
                    x[None, None],
                    architecture=IdentityWithParams(1, 1, kernel_size=3),
                    amp=amp,
                ).astype('float32', copy=False)[0][0]
            )

    def time_patches_grid(self, amp):
        self.predict(self.inp)

    def peakmem_patches_grid(self, amp):
        self.predict(self.inp)
