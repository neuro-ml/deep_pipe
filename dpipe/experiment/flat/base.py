import os
import shutil

import numpy as np


def build_flat_structure(split, config_path, experiment_path, *, makefile):
    makefile_path = os.path.join(os.path.dirname(__file__), makefile)
    assert os.path.exists(makefile_path), f'no {makefile_path} found'

    for i, ids in enumerate(split):
        local = os.path.join(experiment_path, f'experiment_{i}')
        os.makedirs(local)
        shutil.copyfile(makefile_path, os.path.join(local, 'Snakefile'))

        for val, prefix in zip(ids, ('train', 'val', 'test')):
            file = os.path.join(local, f'{prefix}_ids')
            np.savetxt(file, val, delimiter='\n', fmt='%s')
    shutil.copyfile(config_path, os.path.join(experiment_path, 'config.json'))
