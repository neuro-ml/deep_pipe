from abc import abstractmethod, ABC
from typing import Tuple

import numpy as np


class Dataset(ABC):
    """Interface for datasets containing images."""

    ids: Tuple[str] = ()
    n_chans_image: int = None

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

    n_classes: int = None

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
