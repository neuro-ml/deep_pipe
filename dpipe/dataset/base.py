from abc import ABC, abstractmethod
from typing import Sequence, Union


class DataSet(ABC):
    @property
    @abstractmethod
    def ids(self) -> Sequence[Union[str, int]]:
        pass

    @abstractmethod
    def load_x(self, identifier: Union[str, int]):
        pass
