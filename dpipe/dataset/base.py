from abc import abstractmethod, ABC
from typing import Tuple

import numpy as np


class Dataset(ABC):
    ids: Tuple[str] = ()

    @abstractmethod
    def load_image(self, identifier: str) -> np.ndarray:
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

    @property
    @abstractmethod
    def n_chans_image(self) -> int:
        """
        The number of channels in the input image's tensor

        Returns
        -------
        channels: int
        """


class SegmentationDataset(Dataset):
    """Abstract class that describes a dataset for medical image segmentation with labels for pixel."""

    @abstractmethod
    def load_segm(self, identifier: str) -> np.ndarray:
        """
        Load the ground truth segmentation.

        Parameters
        ----------
        identifier: str
            the object's identifier

        Returns
        -------
        segmentation: tensor
            the ground truth segmentation.
        """


class ClassificationDataset(Dataset):
    """Abstract class that describes a dataset for classification."""

    @abstractmethod
    def load_label(self, identifier: str) -> int:
        """
        Loads a dataset entry's label given its identifier

        Parameters
        ----------
        identifier: str
            object's identifier

        Returns
        -------
        label: int
            The entry's label corresponding to the identifier
        """

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """
        Returns
        -------
        num_classes: int
            the number of unique classes present in the dataset
        """
