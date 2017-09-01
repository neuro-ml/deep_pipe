import os

import numpy as np
from tqdm import tqdm

from dpipe.config import get_config, get_resource_manager

if __name__ == '__main__':
    resource_manager = get_resource_manager(
        get_config('config_path', 'input_path', 'output_path', 'transform')
    )

    transform = resource_manager['transform']
    input_path = resource_manager['input_path']
    output_path = resource_manager['output_path']

    transform = resource_manager[resource_manager['transform']]

    os.makedirs(output_path)

    for f in tqdm(os.listdir(input_path)):
        np.save(os.path.join(output_path, f),
                transform(np.load(os.path.join(input_path, f))))
