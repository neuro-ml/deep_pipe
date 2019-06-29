import os
import shutil
import pickle
from pathlib import Path

import torch

__all__ = 'CheckpointManager',


def save_pickle(o, path):
    with open(path, 'wb') as file:
        # TODO: in case of policies __dict__ is an overkill
        pickle.dump(o.__dict__, file)


def load_pickle(o, path):
    with open(path, 'rb') as file:
        state = pickle.load(file)
        for key, value in state.items():
            setattr(o, key, value)


def save_torch(o, path):
    torch.save(o.state_dict(), path)


# TODO: no CPU - GPU interchange for now
def load_torch(o, path):
    o.load_state_dict(torch.load(path))


class CheckpointManager:
    """
    Saves the most recent iteration to ``base_path`` and removes the previous one.

    Parameters
    ----------
    base_path: str
        path to save/restore checkpoint object in/from.
    pickled_objects: dict, None, optional
        objects that will be saved using `pickle`
    state_dict_objects: dict, None, optional
        objects whose ``state_dict()`` will be saved using `torch.save`.
    """

    def __init__(self, base_path: str, pickled_objects: dict = None, state_dict_objects: dict = None):
        self.base_path = Path(base_path)
        self._checkpoint_prefix = 'checkpoint_'
        pickled_objects = pickled_objects or {}
        state_dict_objects = state_dict_objects or {}
        assert not (set(state_dict_objects) & set(pickled_objects))

        self.objects = [(load_pickle, save_pickle, path, o) for path, o in pickled_objects.items()]
        self.objects.extend((load_torch, save_torch, path, o) for path, o in state_dict_objects.items())

    def _get_previous_folder(self, iteration):
        return self.base_path / f'{self._checkpoint_prefix}{iteration - 1}'

    def _get_current_folder(self, iteration):
        return self.base_path / f'{self._checkpoint_prefix}{iteration}'

    def _clear_previous(self, iteration):
        shutil.rmtree(self._get_previous_folder(iteration))

    def save(self, iteration: int):
        """Save the states of all tracked objects."""
        current_folder = self._get_current_folder(iteration)
        os.makedirs(current_folder)

        for _, save, relative_path, o in self.objects:
            save(o, current_folder / relative_path)

        if iteration:
            self._clear_previous(iteration)

    def restore(self) -> int:
        """Restore the most recent states of all tracked objects and return the corresponding iteration."""
        if not self.base_path.exists():
            return 0

        max_iteration = -1
        for file in os.listdir(self.base_path):
            if file.startswith(self._checkpoint_prefix):
                max_iteration = max(max_iteration, int(file[len(self._checkpoint_prefix):]))

        # no backups found
        if max_iteration < 0:
            return 0

        iteration = max_iteration + 1
        last_folder = self._get_previous_folder(iteration)

        for load, _, relative_path, o in self.objects:
            load(o, last_folder / relative_path)

        return iteration
