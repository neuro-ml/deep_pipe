import numpy as np
from torch import nn

from dpipe.predict import patches_grid
from dpipe.torch.model import inference_step


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
        self.predict = patches_grid(200, 100, axis=-1)(
            lambda x: inference_step(x, architecture=nn.Identity(), amp=amp).astype('float32', copy=False)
        )

    def time_patches_grid(self, amp):
        self.predict(self.inp)

    def peakmem_patches_grid(self, amp):
        self.predict(self.inp)
