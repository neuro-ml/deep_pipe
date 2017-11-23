from abc import abstractmethod

import numpy as np


class Segmentation:
    """
    Abstract class that describes a datatset for
    medical image segmentation
    """

    def load_image(self, identifier: str) -> np.array:
        """
        Loads a dataset entry given its identifier

        Parameters
        ----------
        identifier: str
            object's identifier

        Returns
        -------
        object:
            The entry corresponding to the identifier
        """

    @abstractmethod
    def load_segm(self, identifier: str) -> np.array:
        """
        Load the ground truth segmentation

        Parameters
        ----------
        identifier: str
            the object's identifier

        Returns
        -------
        segmentation: integer tensor
            the ground truth segmentation as an integer tensor.
            Each value must correspond to a class.
        """

    @abstractmethod
    def load_msegm(self, identifier) -> np.array:
        """
        Load the multimodal ground truth segmentation

        Parameters
        ----------
        identifier: str
            the object's identifier

        Returns
        -------
        segmentation: bool tensor
            the ground truth segmentation as a bool tensor.
            Each channel must correspond to a class.
        """

    @property
    @abstractmethod
    def n_chans_segm(self):
        """
        The number of channels in the segmentation tensor

        Returns
        -------
        channels: int
        """

    @property
    @abstractmethod
    def n_chans_msegm(self):
        """
        The number of channels in the multimodal segmentation tensor

        Returns
        -------
        channels: int
        """

    @property
    @abstractmethod
    def n_chans_image(self) -> int:
        """
        The number of channels in the input image's tensor
        
        Returns
        -------
        channels: int
        """
