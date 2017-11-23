import os
import json
import shutil
from typing import Iterable

from dpipe.config import get_resource_manager


def flat(split: Iterable, config_path: str, experiment_path: str, *, makefile: str):
    """
    Generates and experiment with a 'flat' structure: each created subdirectory
    will contain triplets of ids (train, validation, test) defined in the `split`.

    Parameters
    ----------
    split: Iterable
        an iterable that yield triplets of ids: (train, validation, test)
    config_path: str
        the path to the config file
    experiment_path: str
        the path where the experiment will be stored
    makefile: str
        the path to the Snakemake file with experiment instructions
    """
    makefile_path = os.path.join(os.path.dirname(__file__), makefile)
    assert os.path.exists(makefile_path), f'no {makefile_path} found'

    for i, ids in enumerate(split):
        local = os.path.join(experiment_path, f'experiment_{i}')
        os.makedirs(local)
        shutil.copyfile(makefile_path, os.path.join(local, 'Snakefile'))

        for val, prefix in zip(ids, ('train', 'val', 'test')):
            path = os.path.join(local, f'{prefix}_ids.json')
            with open(path, "w") as f:
                json.dump(val, f, indent=0)

    # resource manager is needed here, because there may be inheritance
    rm = get_resource_manager(config_path)
    rm.save_config(os.path.join(experiment_path, 'config'))
