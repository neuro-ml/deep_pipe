import numpy as np

from dpipe.predict import patches_grid


class TimeSuite:
    params = [('float16', 'float32', 'float64')]
    timeout = 300

    def setup(self, dtype):
        self.inp = np.random.randn(512, 512, 512).astype(dtype)
        self.predict = patches_grid(200, 100, axis=-1)(lambda x: x)


    def time_patches_grid(self, dtype):
        self.predict(self.inp)

    def peakmem_patches_grid(self, dtype):
        self.predict(self.inp)
