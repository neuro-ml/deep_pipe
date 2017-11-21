from abc import ABC, abstractmethod


class BatchPredict(ABC):
    """
    Interface that realizes the validation and inference logic.
    """

    @abstractmethod
    def validate(self, x, y, *, validate_fn):
        """
        Realizes the validation logic.

        Parameters
        ----------
        x:
            a single input object
        y:
            a single ground truth object
        validate_fn: callable(x, y) -> prediction, loss
            callable, that receives an input batch and a ground truth batch
            and returns the prediction batch and the loss

        Returns
        -------
        prediction:
            prediction for the input
        loss: float
            the validation loss
        """

    @abstractmethod
    def predict(self, x, *, predict_fn):
        """
        Realizes the inference logic.

        Parameters
        ----------
        x:
            a single input object
        predict_fn: callable(x) -> prediction
            callable, that receives an input batch
            and returns the prediction batch

        Returns
        -------
        prediction:
            prediction for the input
        """
