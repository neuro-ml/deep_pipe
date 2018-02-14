from abc import abstractmethod, ABC

import numpy as np


class Dataset(ABC):
    @property
    @abstractmethod
    def ids(self):
        """Returns a tuple of ids of all objects in the dataset."""

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
    """Abstract class that describes a dataset for medical image segmentation."""

    @abstractmethod
    def load_segm(self, identifier: str) -> np.ndarray:
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
    def load_msegm(self, identifier: str) -> np.ndarray:
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
    def n_chans_segm(self) -> int:
        """
        The number of channels in the segmentation tensor

        Returns
        -------
        channels: int
        """

    @property
    @abstractmethod
    def n_chans_msegm(self) -> int:
        """
        The number of channels in the multimodal segmentation tensor

        Returns
        -------
        channels: int
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
