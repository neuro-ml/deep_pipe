import os
from abc import ABC, abstractmethod


def get_model_path(path):
    return os.path.join(path, 'model')


class Model(ABC):
    @abstractmethod
    def do_train_step(self, *inputs, lr):
        pass

    @abstractmethod
    def do_val_step(self, *inputs):
        pass

    @abstractmethod
    def do_inf_step(self, *inputs):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass


class FrozenModel(ABC):
    @abstractmethod
    def do_inf_step(self, *inputs):
        pass
