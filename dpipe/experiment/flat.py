import os
from pathlib import Path
from typing import Iterable, Sequence

from ..config import get_resource_manager
from ..medim.io import PathLike, dump_json


def flat(split: Iterable[Sequence], config_path: PathLike, experiment_path: PathLike,
         prefixes: Sequence[str] = ('train', 'val', 'test')):
    """
    Generates an experiment with a 'flat' structure.
    For each entry of ``split`` a subdirectory of ``experiment_path``
    containing the corresponding identifiers is created.

    Also, the config file from ``config_path`` is copied to ``experiment_path/resources.config``.

    Parameters
    ----------
    split
        an iterable with groups of ids.
    config_path
        the path to the config file.
    experiment_path
        the path where the experiment will be created.
    prefixes
        the corresponding prefixes for each identifier group of ``split``.

    Examples
    --------
    >>> ids = [
    >>>     [[1, 2, 3], [4, 5, 6], [7, 8]],
    >>>     [[1, 4, 8], [7, 5, 2], [6, 3]],
    >>> ]
    >>> flat(ids, 'some_path.config', 'experiments/base')
    # resulting folder structure:
    # experiments/base:
    #   - resources.config
    #   - experiment_0:
    #       - train_ids.json # 1, 2, 3
    #       - val_ids.json # 4, 5, 6
    #       - test_ids.json # 7, 8
    #   - experiment_1:
    #       - train_ids.json # 1, 4, 8
    #       - val_ids.json # 7, 5, 2
    #       - test_ids.json # 6, 3
    """
    experiment_path = Path(experiment_path)
    for i, ids in enumerate(split):
        if len(ids) != len(prefixes):
            raise ValueError(f"The number of identifier groups ({len(ids)}) "
                             f"does not match the number of prefixes ({len(prefixes)})")

        local = experiment_path / f'experiment_{i}'
        os.makedirs(local)

        for val, prefix in zip(ids, prefixes):
            dump_json(val, local / f'{prefix}_ids.json', indent=0)

    # resource manager is needed here, because there may be inheritance
    get_resource_manager(config_path).save_config(experiment_path / 'resources.config')
