import numpy as np

from dpipe.config import register
from .from_metadata import FromMetadata


# We need this class because in the original data segm values are [0, 1, 2, 4]
@register('brats2017')
class Brats2017(FromMetadata):
    def __init__(self, data_path, metadata_rpath='metadata.csv'):
        super().__init__(
            data_path=data_path,
            metadata_rpath=metadata_rpath,
            modalities=['t1', 't1ce', 't2', 'flair'],
            target='segm',
            segm2msegm_matrix=np.array([
                [0, 0, 0],
                [1, 1, 0],
                [1, 0, 0],
                [1, 1, 1]
            ], dtype=bool)
        )

    def load_segm(self, patient_id):
        segm = super().load_segm(patient_id)
        segm[segm == 4] = 3
        return segm

# For Brats 2015
# segm2msegm = np.array([
#     [0, 0, 0],
#     [1, 1, 0],
#     [1, 0, 0],
#     [1, 1, 0],
#     [1, 1, 1]
# ], dtype=bool)
