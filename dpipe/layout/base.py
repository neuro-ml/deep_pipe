import contextlib
import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence, Iterable

import lazycon

from ..io import save, PathLike, load


class Layout(ABC):
    @abstractmethod
    def build(self, *args, **kwargs):
        """Build a new experiment."""

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run a built experiment."""

    @abstractmethod
    def build_parser(self, parser: ArgumentParser):
        """Appropriately updates a console arguments parser for the `build` method."""

    @abstractmethod
    def run_parser(self, parser: ArgumentParser):
        """Appropriately updates a console arguments parser for the `run` method."""


@contextlib.contextmanager
def change_current_dir(folder: PathLike):
    current = os.getcwd()
    try:
        os.chdir(folder)
        yield
    finally:
        os.chdir(current)


class Flat(Layout):
    """
    Generates an experiment with a 'flat' structure.
    Creates a subdirectory of ``experiment_path`` for the each entry of ``split``.
    The subdirectory contains corresponding structure of identifiers.

    Also, the config file from ``config_path`` is copied to ``experiment_path/resources.config``.

    Parameters
    ----------
    split
        an iterable with groups of ids.
    prefixes: Sequence[str]
        the corresponding prefixes for each identifier group of ``split``
        which will be used to generate appropriate filenames.
        Default is ``('train', 'val', 'test')``.

    Examples
    --------
    >>> ids = [
    >>>     [[1, 2, 3], [4, 5, 6], [7, 8]],
    >>>     [[1, 4, 8], [7, 5, 2], [6, 3]],
    >>> ]
    >>> Flat(ids).build('some_path.config', 'experiments/base')
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

    def __init__(self, split: Iterable[Sequence], prefixes: Sequence[str] = ('train', 'val', 'test')):
        self.prefixes = prefixes
        self.split = list(split)

    @staticmethod
    def _expand_prefix(prefix):
        return f'{prefix}_ids.json'

    def build(self, config: PathLike, folder: PathLike, keep: Sequence[str] = None):
        folder = Path(folder)
        for i, ids in enumerate(self.split):
            # TODO: move check to constructor
            if len(ids) != len(self.prefixes):
                raise ValueError(f"The number of identifier groups ({len(ids)}) "
                                 f"does not match the number of prefixes ({len(self.prefixes)})")

            local = folder / f'experiment_{i}'
            local.mkdir(parents=True)

            for val, prefix in zip(ids, self.prefixes):
                save(val, local / self._expand_prefix(prefix), indent=0)

        # resource manager is needed here, because there may be inheritance
        lazycon.load(config).dump(folder / 'resources.config', keep)

    def run(self, config: PathLike, folds: Sequence[int] = None):
        root = Path(config).parent
        if folds is None:
            folds = range(len(self.split))

        for fold in folds:
            with change_current_dir(root / f'experiment_{fold}'):
                lazycon.load(config).get('run_experiment')

    def build_parser(self, parser: ArgumentParser):
        parser.add_argument('folder', help='Destination folder.')
        parser.add_argument('--keep', nargs='+', default=None, help='The definitions to keep. By default - all.')

    def run_parser(self, parser: ArgumentParser):
        parser.add_argument('-f', '--folds', nargs='+', help='Folds to run.')

    def get_ids(self, prefix, folder='.'):
        assert prefix in self.prefixes
        return load(Path(folder) / self._expand_prefix(prefix))

    def __getattr__(self, prefix):
        return self.get_ids(prefix, '.')
