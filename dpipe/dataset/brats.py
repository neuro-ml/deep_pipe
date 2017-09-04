import functools

import numpy as np

from .from_metadata import FromMetadata


Brats2017 = functools.partial(
    FromMetadata,
    metadata_rpath='metadata.csv',
    modalities=['T1', 'T1CE', 'T2', 'flair'],
    target='segm',
    segm2msegm_matrix=np.array(
        [
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1]
        ], dtype=bool
    )
)

# For Brats 2015
# segm2msegm = np.array([
#     [0, 0, 0],
#     [1, 1, 0],
#     [1, 0, 0],
#     [1, 1, 0],
#     [1, 1, 1]
# ], dtype=bool)
