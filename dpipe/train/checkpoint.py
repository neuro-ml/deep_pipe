import os
import shutil
import pickle
from abc import abstractmethod
from pathlib import Path

import torch


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


class CheckpointManagerBase:
    @abstractmethod
    def save(self, iteration: int):
        """Save the states of all tracked objects."""

    @abstractmethod
    def restore(self) -> int:
        """
        Restore the most recent states of all tracked objects and return
        the corresponding iteration.
        """


class DummyCheckpointManager:
    def save(self, iteration: int):
        pass

    @staticmethod
    def restore():
        return 0


class CheckpointManager:
    def __init__(self, base_path: str, pickled_objects: dict = None, state_dict_objects: dict = None):
        self.base_path = Path(base_path)
        self._checkpoint_prefix = 'checkpoint_'
        self.pickled_objects = pickled_objects or {}
        self.state_dict_objects = state_dict_objects or {}

        assert not (set(self.state_dict_objects) & set(self.pickled_objects))

    def _get_previous_folder(self, iteration):
        return self.base_path / f'{self._checkpoint_prefix}{iteration - 1}'

    def _get_current_folder(self, iteration):
        return self.base_path / f'{self._checkpoint_prefix}{iteration}'

    def _clear_previous(self, iteration):
        shutil.rmtree(self._get_previous_folder(iteration))

    def save(self, iteration: int):
        current_folder = self._get_current_folder(iteration)
        os.makedirs(current_folder)

        # TODO: generalize
        for relative_path, o in self.pickled_objects.items():
            save_pickle(o, current_folder / relative_path)

        for relative_path, o in self.state_dict_objects.items():
            save_torch(o, current_folder / relative_path)

        if iteration:
            self._clear_previous(iteration)

    def restore(self):
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

        # TODO: generalize
        for relative_path, o in self.pickled_objects.items():
            load_pickle(o, last_folder / relative_path)

        for relative_path, o in self.state_dict_objects.items():
            load_torch(o, last_folder / relative_path)

        return iteration
