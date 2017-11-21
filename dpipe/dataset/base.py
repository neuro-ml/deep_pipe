from abc import ABC, abstractmethod
from typing import Sequence, Union


class DataSet(ABC):
    """
    DataSet interface
    """

    @property
    @abstractmethod
    def ids(self) -> Sequence[Union[str, int]]:
        """
        Returns
        -------
        ids: Sequence of str or int
            the ids of all the objects in the dataset
        """

    @abstractmethod
    def load_x(self, identifier: Union[str, int]):
        """
        Loads a dataset entry given its identifier

        Parameters
        ----------
        identifier: int, str
            object's identifier

        Returns
        -------
        object:
            The entry corresponding to the identifier
        """
