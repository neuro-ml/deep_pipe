import numpy as np

from .csv import CSV, multiple_columns
from .base import IntSegmentationDataset, SegmentationDataset


class FromCSVMultiple(CSV, SegmentationDataset):
    def __init__(self, data_path, modalities, targets, metadata_rpath):
        super().__init__(data_path, metadata_rpath)
        self.modality_cols = modalities
        self.target_cols = targets
        self.n_chans_msegm = len(modalities)
        self.n_chans_segm = len(modalities)

    @property
    def n_chans_image(self):
        return len(self.modality_cols)

    def load_image(self, identifier):
        return multiple_columns(self.load, identifier, self.modality_cols)

    def load_segm(self, identifier) -> np.array:
        return multiple_columns(self.load, identifier, self.target_cols)


class FromCSVInt(CSV, IntSegmentationDataset):
    def __init__(self, data_path, modalities, target, metadata_rpath, segm2msegm_matrix):
        super().__init__(data_path, metadata_rpath)
        assert type(target) is str
        self.target_col = target
        self.modality_cols = modalities

        assert segm2msegm_matrix.dtype == bool
        self._segm2msegm_matrix = np.array(segm2msegm_matrix)

    @property
    def n_chans_image(self):
        return len(self.modality_cols)

    def load_image(self, identifier):
        return multiple_columns(self.load, identifier, self.modality_cols)

    @property
    def segm2msegm_matrix(self) -> np.array:
        return self._segm2msegm_matrix

    def load_segm(self, identifier):
        return self.load(identifier, self.target_col)

    def segm2msegm(self, x) -> np.array:
        assert np.issubdtype(x.dtype, np.integer), \
            f'Segmentation dtype must be int, but {x.dtype} provided'
        return np.rollaxis(self.segm2msegm_matrix[x], -1)

    def load_msegm(self, identifier) -> np.array:
        """"Method returns multimodal segmentation of shape
         [n_chans_msegm, x, y, z]. We use this result to compute dice scores"""
        return self.segm2msegm(self.load_segm(identifier))

    @property
    def n_chans_segm(self):
        return self.segm2msegm_matrix.shape[0]

    @property
    def n_chans_msegm(self):
        return self.segm2msegm_matrix.shape[1]
