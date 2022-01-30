import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Union

import torch
from dpipe.im.utils import composition
from dpipe.io import PathLike

__all__ = 'Checkpoints', 'CheckpointManager'


def save_pickle(o, path: PathLike):
    if hasattr(o, '__getstate__'):
        state = o.__getstate__()
    else:
        state = o.__dict__

    with open(path, 'wb') as file:
        pickle.dump(state, file)


def load_pickle(o, path: PathLike):
    with open(path, 'rb') as file:
        state = pickle.load(file)

    if hasattr(o, '__setstate__'):
        o.__setstate__(state)
    else:
        for key, value in state.items():
            setattr(o, key, value)


def save_torch(o, path: PathLike):
    torch.save(o.state_dict(), path)


def load_torch(o, path: PathLike):
    o.load_state_dict(torch.load(path))


class Checkpoints:
    """
    Saves the most recent iteration to ``base_path`` and removes the previous one.

    Parameters
    ----------
    base_path: str
        path to save/restore checkpoint object in/from.
    objects: Dict[PathLike, Any]
        objects to save. Each key-value pair represents
        the path relative to ``base_path`` and the corresponding object.
    frequency: int
        the frequency with which the objects are stored.
        By default only the latest checkpoint is saved.
    """

    def __init__(self, base_path: PathLike, objects: Union[Iterable, Dict[PathLike, Any]], frequency: int = None):
        self.base_path: Path = Path(base_path)
        self._checkpoint_prefix = 'checkpoint_'
        if not isinstance(objects, dict):
            objects = self._generate_unique_names(objects)
        self.objects = objects or {}
        self.frequency = frequency or float('inf')

    @staticmethod
    @composition(dict)
    def _generate_unique_names(objects):
        names = set()
        for o in objects:
            name = type(o).__name__
            if name in names:
                idx = 1
                while f'{name}_{idx}' in names:
                    idx += 1

                name = f'{name}_{idx}'

            assert name not in names
            names.add(name)
            yield name, o

    def _get_checkpoint_folder(self, iteration: int):
        return self.base_path / f'{self._checkpoint_prefix}{iteration}'

    def _clear_checkpoint(self, iteration: int):
        if (iteration + 1) % self.frequency != 0:
            shutil.rmtree(self._get_checkpoint_folder(iteration))

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

    def _save_to(self, folder: Path):
        for path, o in self.objects.items():
            save = self._dispatch_saver(o)
            save(o, folder / path)

    def save(self, iteration: int, train_losses: Sequence = None, metrics: dict = None):
        """Save the states of all tracked objects."""
        current_folder = self._get_checkpoint_folder(iteration)
        current_folder.mkdir(parents=True)
        self._save_to(current_folder)

        if iteration:
            self._clear_checkpoint(iteration - 1)

    def restore(self) -> int:
        """Restore the most recent states of all tracked objects and return next iteration's index."""
        if not self.base_path.exists():
            return 0

        max_iteration = -1
        for file in self.base_path.iterdir():
            filename = file.name
            if filename.startswith(self._checkpoint_prefix):
                max_iteration = max(max_iteration, int(filename[len(self._checkpoint_prefix):]))

        # no backups found
        if max_iteration < 0:
            return 0

        iteration = max_iteration + 1
        last_folder = self._get_checkpoint_folder(iteration - 1)

        for path, o in self.objects.items():
            load = self._dispatch_loader(o)
            load(o, last_folder / path)

        return iteration


CheckpointManager = Checkpoints
