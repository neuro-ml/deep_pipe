from abc import ABC, abstractmethod

import numpy as np
from typing import List


class Dataset(ABC):
    @abstractmethod
    def __init__(self, processed_path):
        pass

    @abstractmethod
    def load_mscan(self, patient_id) -> np.array:
        pass

    @abstractmethod
    def load_segm(self, patient_id) -> np.array:
        pass

    @abstractmethod
    def load_msegm(self, patient) -> np.array:
        pass

    @abstractmethod
    def segm2msegm(self, segm) -> np.array:
        pass

    @property
    @abstractmethod
    def patient_ids(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def n_chans_mscan(self) -> int:
        pass

    @property
    @abstractmethod
    def n_chans_msegm(self) -> int:
       pass

    @property
    @abstractmethod
    def n_classes(self) -> int:
        pass

    @property
    @abstractmethod
    def spatial_size(self) -> List[int]:
        pass
