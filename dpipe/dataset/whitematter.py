import numpy as np
from .from_csv import FromCSVInt
from dpipe.config import register


@register('wmh2017')
class Wmh2017(FromCSVInt):
    def __init__(self, data_path, modalities, target):
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
