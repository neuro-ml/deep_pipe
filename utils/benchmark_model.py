import os
import time
import argparse

import numpy as np
import torch.nn as nn

from dpipe.config import get_resource_manager


shape = (1, 190, 217, 154)
n_samples = 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_known_args()[0]
    rm = get_resource_manager(args.config_path)

    model_core: nn.Module = rm.model_core
    predict = rm.predict

    def sample():
        return np.random.randn(*shape).astype(np.float32)


    for cuda in (True, False):
        if cuda:
            model_core.cuda()
        else:
            model_core.cpu()

        # To initiate

        for i in range(2):
            predict(sample())

        total_time = 0.0

        for i in range(n_samples):
            x = sample()
            start_time = time.time()
            predict(x)
            finish_time = time.time()

            total_time += finish_time - start_time

        time_per_sample = total_time / n_samples

        print(f'GPU: {cuda}, time: {time_per_sample:.4f}')
