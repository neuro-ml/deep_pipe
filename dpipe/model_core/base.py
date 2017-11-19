from abc import ABC, abstractmethod


class ModelCore(ABC):
    """
    Interface for the tensorflow's computational graphs.
    """

    def __init__(self, n_chans_in: int, n_chans_out: int):
        """
        Parameters
        ----------
        n_chans_in: int
            number of input channels
        n_chans_out: int
            number of output channels
        """
        self.n_chans_in = n_chans_in
        self.n_chans_out = n_chans_out

    @abstractmethod
    def build(self, training_ph):
        """
        Build the computational graph along with placeholders and operations.

        Parameters
        ----------
        training_ph: boolean tensor placeholder
            denotes the network's state (training/evaluation)

        Returns
        -------
        x_phs: Sequence of placeholders
            The placeholders for the network's inputs
        logits: placeholder
            placeholder for the network's output.
            The leading two dimensions must be batch_size and n_chans_out:
                [batch_size, n_chans_out, spatial_dims...].
        """
