from abc import ABC, abstractmethod

from ..optimizer import Optimizer


class ModelCore(ABC):
    def __init__(self, optimizer: Optimizer, n_chans_in, n_chans_out):
        self.optimizer = optimizer
        self.n_chans_in = n_chans_in
        self.n_chans_out = n_chans_out

        # Gets initialized during build
        self.train_input_phs = self.inference_input_phs = None
        self.loss = self.y_pred = None

    @abstractmethod
    def build(self, training):
        """Method defines placeholders and tensors, necessary for the model."""
        pass

    @abstractmethod
    def validate_object(self, x, y, do_val_step):
        pass

    @abstractmethod
    def predict_object(self, x, do_inf_step):
        pass
