import os
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
from dpipe.io import PathLike

from dpipe.checks import join
from dpipe.commands import lock_dir as _lock


def _path_based_call(exists, missing, exists_message, missing_message, paths, keyword_paths, verbose):
    outputs = paths + tuple(keyword_paths.values())
    if not outputs:
        raise ValueError('At least one path must be provided either via positional or keyword arguments.')

    def _print_message(message):
        if message and verbose:
            print(f'\n>>> {message}: {join(outputs)}\n', flush=True)

    if all(map(os.path.exists, outputs)):
        _print_message(exists_message)
        return exists(*paths, **keyword_paths)

    _print_message(missing_message)
    try:
        value = missing(*paths, **keyword_paths)
    except BaseException as e:
        list(map(shutil.rmtree, filter(os.path.exists, outputs)))
        raise RuntimeError('An exception occurred. The outputs were cleaned up.\n') from e

    missing_paths = [path for path in outputs if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError(f'The following outputs were not generated: {join(missing_paths)}')

    return value


@np.deprecate
def if_missing(func: Callable, *paths: str, verbose: bool = True, **keyword_paths: str):
    """
    Call ``func`` if at least some of the ``paths`` or ``keyword_paths`` do not exist.

    Examples
    --------
    >>> if_missing(save_results, 'values', 'metrics', misc='temp_data')
    # if `values` or `metrics` or `temp_data` do not exist, the following call will be performed:
    >>> save_results('values', 'metrics', misc='temp_data')
    """
    _path_based_call(
        lambda *x, **y: None, func,
        'Nothing to be done, all outputs already exist', 'Running command to generate outputs',
        paths, keyword_paths, verbose
    )


def run(*args):
    """Returns the last argument. Useful in config files."""
    if not args:
        raise ValueError('Nothing to run.')
    return args[-1]


lock_dir = np.deprecate(_lock, message='lock_dir was moved to `dpipe.commands`')


class Locker:
    def __init__(self, folder: PathLike = '.', lock: str = '.lock'):
        lock = Path(folder) / lock
        if lock.exists():
            raise FileExistsError(f'Trying to lock directory {lock.resolve().parent}, but it is already locked.')

        lock.touch(exist_ok=False)
        self.lock = lock

    def run(*args):
        if not args:
            raise AttributeError('Use "Locker().run instead of Locker.run"')
        self, *args = args
        if not isinstance(self, Locker):
            raise AttributeError('Use "Locker().run instead of Locker.run"')

        os.remove(self.lock)

    def __del__(self):
        if hasattr(self, 'lock') and self.lock.exists():
            os.remove(self.lock)
