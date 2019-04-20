import time
import argparse

import numpy as np
import torch.nn as nn
import gpustat
from tqdm import tqdm

from dpipe.config import get_resource_manager


shape = (1, 190, 217, 154)
n_samples = 5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('--real_examples', help='allows to calculate on real dataset examples', action='store_true')
    parser.add_argument('--lowres', help='compatibility with lowres `models_core`', action='store_true')
    args = parser.parse_known_args()[0]
    rm = get_resource_manager(args.config_path)

    pth = args.config_path.strip('resources.config')
    pth += 'experiment_0/model.pth'
    model = rm.model
    model.load(pth)
    if args.lowres:
        model_core: nn.Module = model.models_core
    else:
        model_core: nn.Module = model.model_core
    predict = rm.predict

    def sample(real_examples=False):
        if real_examples:
            ids = rm.split[0][2]  # + rm.split[1][2]  # hardcoded test ids for 2 splits we use
            _id = np.random.choice(ids)
            return rm.load_x(_id)
        else:
            return np.random.randn(*shape).astype(np.float32)

    for cuda in (False, True):
        if cuda:
            model_core.cuda()
        else:
            model_core.cpu()

        # To initiate

        for i in range(2):
            predict(sample())

        total_time = []

        for i in tqdm(range(n_samples)):
            x = sample(real_examples=args.real_examples)
            start_time = time.time()
            predict(x)
            finish_time = time.time()

            total_time.append(finish_time - start_time)

        avg_time = np.mean(total_time)
        std_time = np.std(total_time)

        print(f'GPU: {cuda}, time: {avg_time:.4f} +- {std_time:.4f}')
        if cuda:
            for gpu in gpustat.new_query():
                print(f'used: {gpu.memory_used} total: {gpu.memory_total}')
