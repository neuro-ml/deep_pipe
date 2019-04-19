import os

import numpy as np
import pandas as pd

from dpipe.medim import load_image
from dpipe.medim.io import PathLike
from .base import Dataset


def multiple_columns(method, index, columns):
    return np.array([method(index, col) for col in columns])


class CSV(Dataset):
    """
    A small wrapper for dataframes that contain paths to data.

    Parameters
    ----------
    path
        the path relative to which the data is located.
    filename
        the relative path to the csv dataframe.
    index_col
        the column that will be used as index. Must contain unique values.
    """

    def __init__(self, path: PathLike, filename: str = 'meta.csv', index_col: str = 'id'):
        self.path = path
        self.filename = filename

        df = pd.read_csv(os.path.join(path, filename), encoding='utf-8')
        if index_col is not None:
            df[index_col] = df[index_col].astype(str)
            df = df.set_index(index_col).sort_index()
            if len(df.index.unique()) != len(df):
                raise ValueError(f'The column "{index_col}" doesn\'t contain unique values.')

        self.df: pd.DataFrame = df
        self.ids = tuple(self.df.index)

    def get(self, index, col):
        return self.df.loc[index, col]

    def get_global_path(self, index: str, col: str) -> str:
        """
        Get the global path at ``index`` and ``col``.
        Often data frames contain path to data, this is a convenient way to obtain
        the global path.
        """
        return os.path.join(self.path, self.get(index, col))

    def load(self, index: str, col: str, loader=load_image):
        """Load the value that is located at the path at corresponding ``index`` and ``col``."""
        return loader(self.get_global_path(index, col))

    def __getattr__(self, item):
        return getattr(self.df, item)

    def __getitem__(self, item):
        return self.df[item]
