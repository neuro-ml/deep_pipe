from abc import abstractmethod, ABCMeta
from functools import wraps
from typing import Tuple

import numpy as np


class AbstractAttribute:
    def __init__(self, description: str):
        self.description = description

    def __repr__(self):
        return self.description


class ABCAttributesMeta(ABCMeta):
    def __new__(mcs, *args, **kwargs):
        cls = super().__new__(mcs, *args, **kwargs)
        initialize = cls.__init__

        @wraps(initialize)
        def __init__(self, *args_, **kwargs_):
            return_value = initialize(self, *args_, **kwargs_)

            missing = []
            for name in dir(self):
                value = getattr(self, name)
                if isinstance(value, AbstractAttribute) or value is AbstractAttribute:
                    missing.append(name)
            if missing:
                raise AttributeError(f'Class "{cls.__name__}" requires the following attributes '
                                     f'which are not defined during init: {", ".join(missing)}.')
            return return_value

        cls.__init__ = __init__
        return cls


class Dataset(metaclass=ABCAttributesMeta):
    """Interface for datasets containing images."""

    ids: Tuple[str] = AbstractAttribute
    n_chans_image: int = AbstractAttribute

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

    n_classes: int = AbstractAttribute

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
