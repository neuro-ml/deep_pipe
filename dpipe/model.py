import os
from abc import ABC, abstractmethod


def get_model_path(path):
    return os.path.join(path, 'model')


class Model(ABC):
    """
    Interface for training a neural network model
    """

    @abstractmethod
    def do_train_step(self, *inputs, lr: float) -> float:
        """
        Perform a train step

        Parameters
        ----------
        inputs: Sequence
            a sequence of batches that will be fed into the network
            and used to calculate the loss.
        lr: float
            the learning rate for the gradient step

        Returns
        -------
        loss: float
        """

    @abstractmethod
    def do_val_step(self, *inputs):
        """
        Perform a validation step

        Parameters
        ----------
        inputs: Sequence
            a sequence of batches that will be fed into the network
            and used to calculate the loss.

        Returns
        -------
        prediction: batch
            the prediction for the input batch
        loss: float
        """

    @abstractmethod
    def do_inf_step(self, *inputs):
        """
        Perform an inference step

        Parameters
        ----------
        inputs: Sequence
            a sequence of batches that will be fed into the network

        Returns
        -------
        prediction: batch
            the prediction for the input batch
        """

    @abstractmethod
    def save(self, path: str):
        """
        Save the network parameters

        Parameters
        ----------
        path: str
        """

    @abstractmethod
    def load(self, path):
        """
        Load the network parameters from `path`

        Parameters
        ----------
        path: str
        """


class FrozenModel(ABC):
    """
    Interface for a trained neural network used for inference
    """

    @abstractmethod
    def do_inf_step(self, *inputs):
        """
        Perform an inference step

        Parameters
        ----------
        inputs: Sequence
            a sequence of batches that will be fed into the network

        Returns
        -------
        prediction: batch
            the prediction for the input batch
        """
