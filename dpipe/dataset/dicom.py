import numpy as np

from dpipe.dataset import CSV
from dicom_csv import load_series
from dpipe.io import PathLike


class DICOMDataset(CSV):
    """
    A loader for DICOM series.
    All the metadata is stored at ``filename`` and the DICOM files are located relative to ``path``.

    Parameters
    ----------
    path: PathLike
        the path to the data.
    filename: str
        the relative path to the csv dataframe. Default is ``meta.csv``.
    index_col: str, None, optional
        the column that will be used as index. Must contain unique values. Default is ``id``.

    References
    ----------
    `aggregate_images`, `CSV`
    """

    def __init__(self, path: PathLike, filename: str, index_col: str = 'PatientID'):
        super().__init__(path, filename, index_col)

    def load_image(self, identifier) -> np.ndarray:
        return load_series(self.df.loc[identifier], self.path)
