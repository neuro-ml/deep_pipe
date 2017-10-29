import functools
import numpy as np
from .from_csv import FromCSVInt


class Wmh2017(FromCSVInt):
    def __init__(self, data_path, t1, flair, target, mask=None,
                 mask_value=None):
        self.mask_value = mask_value
        self.mask = mask
        modalities = [t1, flair] if mask is None else [t1, flair, mask]
        super().__init__(
            data_path=data_path,
            metadata_rpath='metadata.csv',
            modalities=modalities,
            target=target,
            segm2msegm_matrix=np.array([[0], [1], [0]], dtype=bool)
        )

    def load_segm(self, patient_id):
        return super().load_segm(patient_id).astype(int)

    @property
    def groups(self):
        return self.dataFrame['cite'].as_matrix()


def cached_property(f):
    return property(functools.lru_cache(1)(f))
