import numpy as np

from dpipe.predict import patches_grid
from imops.utils import imops_num_threads


class TimeSuite:
    params = [('float16', 'float32', 'float64'), list(range(1, 9))]
    timeout = 300

    def setup(self, dtype, num_threads):
        self.inp = np.random.randn(512, 512, 512).astype(dtype)
        self.predict = patches_grid(200, 100, axis=-1)(lambda x: x)


    def time_patches_grid(self, dtype, num_threads):
        with imops_num_threads(num_threads):
            self.predict(self.inp)

    def peakmem_patches_grid(self, dtype, num_threads):
        with imops_num_threads(num_threads):
            self.predict(self.inp)
