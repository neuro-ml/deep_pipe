from typing import Iterable, Sequence

import numpy as np

from ..layout import Flat
from ..io import PathLike


@np.deprecate(message='This function is deprecated in favor of `dpipe.layout.Flat`')
def flat(split: Iterable[Sequence], config_path: PathLike, experiment_path: PathLike,
         prefixes: Sequence[str] = ('train', 'val', 'test')):
    """
    Generates an experiment with a 'flat' structure.
    Creates a subdirectory of ``experiment_path`` for the each entry of ``split``.
    The subdirectory contains corresponding structure of identifiers.

    Also, the config file from ``config_path`` is copied to ``experiment_path/resources.config``.

    Parameters
    ----------
    split: Iterable[Sequence]
        an iterable with groups of ids.
    config_path: PathLike
        the path to the config file.
    experiment_path: PathLike
        the path where the experiment will be created.
    prefixes: Sequence[str]
        the corresponding prefixes for each identifier group of ``split``.
        Default is ``('train', 'val', 'test')``.

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
    Flat(split, prefixes=prefixes).build(config_path, experiment_path)
