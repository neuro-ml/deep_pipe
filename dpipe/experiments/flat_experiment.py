from typing import Iterable
import os
import shutil
import re

import numpy as np

pattern = re.compile('^(.+)_experiment.py$')


def build(split: Iterable, experiment_path: str):
    name = os.path.basename(__file__)
    name = pattern.match(name).group(1)
    base_dir = os.path.dirname(__file__)
    rule = os.path.join(base_dir, name + '.rule')

    for i, ids in enumerate(split):
        local = os.path.join(experiment_path, f'experiment_{i}')
        os.makedirs(local)
        shutil.copyfile(rule, os.path.join(local, 'train_eval.rule'))

        for val, file in zip(ids, ('train', 'eval', 'test')):
            file = os.path.join(local, file + '.ids')
            np.savetxt(file, val, delimiter='\n', fmt='%s')
