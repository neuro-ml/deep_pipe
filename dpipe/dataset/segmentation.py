import numpy as np

from .csv import CSV, multiple_columns
from .base import Dataset


class MultichannelSegmentationFromCSV(CSV, Dataset):
    def __init__(self, data_path, modalities, targets, metadata_rpath, index_col='id'):
        super().__init__(data_path, metadata_rpath, index_col)
        self.modality_cols = modalities
        self.target_cols = targets
        self.n_chans_image = len(self.modality_cols)

    def load_image(self, identifier):
        return multiple_columns(self.load, identifier, self.modality_cols)

    def load_segm(self, identifier):
        return multiple_columns(self.load, identifier, self.target_cols)


class SegmentationFromCSV(MultichannelSegmentationFromCSV):
    def __init__(self, data_path, modalities, target, metadata_rpath):
        super().__init__(data_path, modalities=modalities, targets=[target], metadata_rpath=metadata_rpath)

    def load_segm(self, identifier) -> np.ndarray:
        return super().load_segm(identifier)[0]
