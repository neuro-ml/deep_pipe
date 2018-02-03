import os

import numpy as np
import pandas as pd

from dpipe.medim.utils import load_image


def multiple_columns(method, index, columns):
    return np.array([method(index, col) for col in columns])


class CSV:
    """A small wrapper for csv files."""

    def __init__(self, path, filename='meta.csv', index_col='id'):
        self.path = path
        self.filename = filename

        df = pd.read_csv(os.path.join(path, filename))
        if index_col is not None:
            df[index_col] = df[index_col].astype(str)
            df = df.set_index(index_col).sort_index()
        self.df: pd.DataFrame = df

        self._ids = tuple(self.df.index)

    @property
    def ids(self):
        return self._ids

    def get(self, index, col):
        return self.df.loc[index, col]

    def get_global_path(self, index: str, col: str) -> str:
        """
        Join the slice's result with the data frame's `path`.
        Often data frames contain path to data, this is a convenient way to obtain
        the global path.
        """
        return os.path.join(self.path, self.get(index, col))

    def load(self, index: str, col: str, loader=load_image):
        return loader(self.get_global_path(index, col))
