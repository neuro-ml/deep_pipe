from abc import ABC, abstractmethod
from typing import Sequence

import tensorflow as tf


class ModelCore(ABC):
    def __init__(self, n_chans_in, n_chans_out):
        self.n_chans_in = n_chans_in
        self.n_chans_out = n_chans_out

    @abstractmethod
    def build(self, training_ph) -> (Sequence[tf.placeholder], tf.Tensor):
        """Method returning x_phs as a list and logits"""
        pass

    @abstractmethod
    def validate_object(self, x, y, do_val_step):
        pass

    @abstractmethod
    def predict_object(self, x, do_inf_step):
        pass
