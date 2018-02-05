from typing import Sequence


class LearningRatePolicy:
    """Interface for learning rate policies. lr attribute is a current learning rate."""

    def __init__(self, lr_init):
        self.epoch = 0
        self.step = 0
        self.total_steps = 0
        self.lr = lr_init

    def on_epoch_finished(self, *, train_losses: Sequence[float] = None, val_losses: Sequence[float] = None,
                          metrics: dict = None):
        pass

    def epoch_finished(self, *, train_losses: Sequence[float] = None, val_losses: Sequence[float] = None,
                       metrics: dict = None):
        """Indicate that an epoch has finished, with corresponding losses and metrics."""
        self.on_epoch_finished(train_losses=train_losses, val_losses=val_losses, metrics=metrics)
        self.step = 0
        self.epoch += 1

    def on_step_finished(self, train_loss: float):
        pass

    def step_finished(self, train_loss: float):
        """Indicate that a new training step in the current epoch has finished."""
        self.on_step_finished(train_loss)
        self.step += 1
        self.total_steps += 1
