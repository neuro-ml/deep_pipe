from abc import ABC, abstractmethod
from typing import List

import tensorflow as tf


class Model(ABC):
    @property
    @abstractmethod
    def graph(self) -> tf.Graph:
        return tf.get_default_graph()

    @property
    @abstractmethod
    def x_phs(self) -> List[tf.placeholder]:
        pass

    @property
    @abstractmethod
    def y_phs(self) -> List[tf.placeholder]:
        pass

    @property
    @abstractmethod
    def training_ph(self) -> tf.placeholder:
        pass

    @property
    @abstractmethod
    def loss(self) -> tf.Tensor:
        pass


class SegmentationModel(Model):
    @property
    @abstractmethod
    def y_pred(self) -> tf.Tensor:
        pass