from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class ModelCore(ABC):
    """Abstract model core, most important method is build."""
    def __init__(self, n_chans_in, n_chans_out):
        self.n_chans_in = n_chans_in
        self.n_chans_out = n_chans_out

    @abstractmethod
    def build(self, training_ph) -> ([tf.placeholder], tf.Tensor):
        """Build computational graph: x_phs and operations.
        Returns ([x_phs], logits).
        Logits have shape [batch_size, n_chans_out, spatial_dims...]."""
        pass

    @abstractmethod
    def validate_object(self, x, y, do_val_step) -> (np.array, float):
        pass

    @abstractmethod
    def predict_object(self, x, do_inf_step) -> np.array:
        pass
