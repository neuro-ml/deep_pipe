import os

import numpy as np
from tqdm import tqdm

from dpipe.config import get_args, get_resource_manager

if __name__ == '__main__':
    rm = get_resource_manager(
        **get_args('config_path', 'input_path', 'output_path', 'transform')
    )

    transform = getattr(rm, rm.transform)
    os.makedirs(rm.output_path)

    for f in tqdm(os.listdir(rm.input_path)):
        np.save(os.path.join(rm.output_path, f),
                transform(np.load(os.path.join(rm.input_path, f))))
