from abc import abstractmethod, ABC
from typing import Sequence


class LearningRate(ABC):
    """Interface for learning rate policies"""

    def __init__(self):
        self.epoch = 0
        self.step = 0
        self.total_steps = 0

    def next_epoch(self):
        """Indicate that a new epoch has begun"""
        self.epoch += 1
        self.step = 0

    def next_step(self):
        """Indicate that a new training step in the current epoch has begun"""
        self.step += 1
        self.total_steps += 1

    @abstractmethod
    def next_lr(self, train_losses: Sequence, val_losses: Sequence, metrics: dict) -> float:
        """
        Get the next learning rate.

        Parameters
        ----------
        train_losses: Sequence
        val_losses: Sequence
        metrics: dict
            a dict containing the metrics calculated on the validation set

        Returns
        -------
        learning_rate: float
        """
