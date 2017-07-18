from typing import Iterable
import os
import shutil

import numpy as np


def build(split: Iterable, experiment_dir: str):
    name = '.'.join(os.path.basename(__file__).split('.')[:-1])
    base_dir = os.path.dirname(__file__)
    snake = f'{name}.snake'
    rule_path = os.path.join(base_dir, snake)

    for i, ids in enumerate(split):
        local = os.path.join(experiment_dir, f'experiment_{i}')
        os.makedirs(local)
        shutil.copyfile(rule_path, os.path.join(local, snake))

        for val, file in zip(ids, ('train', 'val', 'test')):
            file = os.path.join(local, f'{file}_ids')
            np.savetxt(file, val, delimiter='\n', fmt='%s')
