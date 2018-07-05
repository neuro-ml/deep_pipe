import os
import json
from typing import Iterable

from dpipe.config import get_resource_manager


def flat(split: Iterable, config_path: str, experiment_path: str):
    """
    Generates an experiment with a 'flat' structure: each created subdirectory
    will contain triples of ids (train, validation, test) defined in the `split`.

    Parameters
    ----------
    split: Iterable
        an iterable that yield triplets of ids: (train, validation, test)
    config_path: str
        the path to the config file
    experiment_path: str
        the path where the experiment will be stored
    """
    for i, ids in enumerate(split):
        local = os.path.join(experiment_path, f'experiment_{i}')
        os.makedirs(local)

        for val, prefix in zip(ids, ('train', 'val', 'test')):
            path = os.path.join(local, f'{prefix}_ids.json')
            with open(path, "w") as f:
                json.dump(val, f, indent=0)

    # resource manager is needed here, because there may be inheritance
    get_resource_manager(config_path).save_config(os.path.join(experiment_path, 'resources.config'))
