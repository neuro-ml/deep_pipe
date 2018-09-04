from abc import ABC, abstractmethod


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
    def load(self, path: str, modify_state_fn: callable):
        """
        Load the network parameters from `path`. Possibly modify them using `modify_state_fn`.

        Parameters
        ----------
        path: str
        modify_state_fn: callable
            if not None, two arguments will be passed to the function: current state of the model and the state loaded
            from the path. This function should modify states as needed and return the final state to load. This
            function can be handy if you want to transfer weights from similar but not completely equal architecture.
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
