import shutil
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any

import torch

from dpipe.medim.io import PathLike

__all__ = 'CheckpointManager',


def save_pickle(o, path):
    with open(path, 'wb') as file:
        pickle.dump(o.__dict__, file)


def load_pickle(o, path):
    with open(path, 'rb') as file:
        state = pickle.load(file)
        for key, value in state.items():
            setattr(o, key, value)


def save_torch(o, path):
    torch.save(o.state_dict(), path)


def load_torch(o, path):
    o.load_state_dict(torch.load(path))


class CheckpointManager:
    """
    Saves the most recent iteration to ``base_path`` and removes the previous one.

    Parameters
    ----------
    base_path: str
        path to save/restore checkpoint object in/from.
    objects: Dict[PathLike, Any]
        objects to save. Each key-value pair represents
        the path relative to ``base_path`` and the corresponding object.

    pickled_objects
        deprecated argument
    state_dict_objects
        deprecated argument
    """

    def __init__(self, base_path: PathLike, objects: Dict[PathLike, Any], pickled_objects=None,
                 state_dict_objects=None):
        self.base_path: Path = Path(base_path)
        self._checkpoint_prefix = 'checkpoint_'
        objects = objects or {}
        state_dict_objects = state_dict_objects or {}
        pickled_objects = pickled_objects or {}

        self.objects = {}
        self.objects.update(objects)

        # TODO: deprecated 10.08.2019
        if pickled_objects or state_dict_objects:
            warnings.warn('`pickled_objects` and `state_dict_objects` arguments are deprecated. Use `objects` instead.',
                          DeprecationWarning)
            assert not (set(state_dict_objects) & set(pickled_objects) & set(objects))
            self.objects.update(state_dict_objects)
            self.objects.update(pickled_objects)

    def _get_previous_folder(self, iteration):
        return self.base_path / f'{self._checkpoint_prefix}{iteration - 1}'

    def _get_current_folder(self, iteration):
        return self.base_path / f'{self._checkpoint_prefix}{iteration}'

    def _clear_previous(self, iteration):
        shutil.rmtree(self._get_previous_folder(iteration))

    @staticmethod
    def _dispatch_saver(o):
        if isinstance(o, (torch.nn.Module, torch.optim.Optimizer)):
            return save_torch
        return save_pickle

    @staticmethod
    def _dispatch_loader(o):
        if isinstance(o, (torch.nn.Module, torch.optim.Optimizer)):
            return load_torch
        return load_pickle

    def save(self, iteration: int):
        """Save the states of all tracked objects."""
        current_folder = self._get_current_folder(iteration)
        current_folder.mkdir(parents=True)

        for path, o in self.objects.items():
            save = self._dispatch_saver(o)
            save(o, current_folder / path)

        if iteration:
            self._clear_previous(iteration)

    def restore(self) -> int:
        """Restore the most recent states of all tracked objects and return the corresponding iteration."""
        if not self.base_path.exists():
            return 0

        max_iteration = -1
        for file in self.base_path.iterdir():
            file = file.name
            if file.startswith(self._checkpoint_prefix):
                max_iteration = max(max_iteration, int(file[len(self._checkpoint_prefix):]))

        # no backups found
        if max_iteration < 0:
            return 0

        iteration = max_iteration + 1
        last_folder = self._get_previous_folder(iteration)

        for path, o in self.objects:
            load = self._dispatch_loader(o)
            load(o, last_folder / path)

        return iteration
