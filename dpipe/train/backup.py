import os
import shutil
from os.path import join as jp
import pickle

import torch

from dpipe.torch import load_model_state


def save_pickle(o, path):
    with open(path, 'wb') as file:
        pickle.dump(o, file)


def load_pickle(o, path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def save_torch(o, path):
    torch.save(o.state_dict(), path)


class Backup:
    def __init__(self, base_path: str, pickled_objects: dict = None, state_dict_objects: dict = None):
        self.base_path = base_path
        self.backup_iteration = 0
        self._backup_prefix = 'backup_'
        self.pickled_objects = pickled_objects or {}
        self.state_dict_objects = state_dict_objects or {}

        assert not (set(self.state_dict_objects) & set(self.pickled_objects))

    def _get_next_folder(self):
        return jp(self.base_path, f'{self._backup_prefix}{self.backup_iteration + 1}')

    def _get_current_folder(self):
        return jp(self.base_path, f'backup_{self.backup_iteration}')

    def save(self):
        next_folder = self._get_next_folder()
        os.makedirs(next_folder)

        # TODO: generalize
        for relative_path, o in self.pickled_objects.items():
            save_pickle(o, jp(next_folder, relative_path))

        for relative_path, o in self.state_dict_objects.items():
            save_torch(o, jp(next_folder, relative_path))

        if self.backup_iteration:
            self.clear_backup()
        self.backup_iteration += 1

    def restore(self):
        max_iteration = -1
        for file in os.listdir(self.base_path):
            if file.startswith(self._backup_prefix):
                max_iteration = max(max_iteration, int(file[len(self._backup_prefix):]))

        # no backups found
        if max_iteration < 0:
            return

        self.backup_iteration = max_iteration
        current_folder = self._get_current_folder()

        # TODO: generalize
        for relative_path, o in self.pickled_objects.items():
            self.pickled_objects[relative_path] = load_pickle(o, jp(current_folder, relative_path))

        for relative_path, o in self.state_dict_objects.items():
            self.state_dict_objects[relative_path] = load_model_state(o, jp(current_folder, relative_path))

    def get_value(self, name):
        if name in self.state_dict_objects:
            return self.state_dict_objects[name]
        return self.pickled_objects[name]

    def clear_backup(self):
        shutil.rmtree(self._get_current_folder())
