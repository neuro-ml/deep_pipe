import os
import functools

import nibabel
import numpy as np

from .base import Dataset


def cached_property(f):
    return property(functools.lru_cache(1)(f))


def get_folders(path):
    return next(os.walk(path))[1]


MODALITIES_POSTFIXES = ['_t1.nii.gz', '_t1ce.nii.gz',
                        '_t2.nii.gz', '_flair.nii.gz']
SEGMENTATION_POSTFIX = '_seg.nii.gz'


def load_modality(patient, data_path, postfix):
    modality_path = os.path.join(data_path, patient, patient + postfix)
    data = nibabel.load(modality_path).get_data()
    return data


class Brats2017(Dataset):
    """Data loader for brats 2017. We have replaced label 4 with 3 during data
    preparation."""

    def __init__(self, data_path):
        self.data_path = data_path

    @cached_property
    def patient_ids(self):
        return tuple(sorted(get_folders(self.data_path)))

    def load_mscan(self, patient):
        scans = [load_modality(patient, self.data_path, p)
                 for p in MODALITIES_POSTFIXES]
        return np.array(scans)

    def load_segm(self, patient):
        segm = load_modality(patient, self.data_path, SEGMENTATION_POSTFIX)
        segm = np.array(segm, dtype=np.uint8)
        segm[segm == 4] = 3
        return segm

    @property
    def n_chans_mscan(self):
        return 4

    @cached_property
    def segm2msegm_matrix(self):
        return np.array([
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1]
        ], dtype=bool)

# For Brats 2015
# segm2msegm = np.array([
#     [0, 0, 0],
#     [1, 1, 0],
#     [1, 0, 0],
#     [1, 1, 0],
#     [1, 1, 1]
# ], dtype=bool)
