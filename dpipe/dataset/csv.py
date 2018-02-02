import os

import numpy as np
import pandas as pd

from dpipe.medim.utils import load_image


class CSV:
    """
    A small wrapper for csv files.
    """

    def __init__(self, path, filename='meta.csv', index_col='id'):
        self.path = path
        self.filename = filename

        df = pd.read_csv(os.path.join(path, filename))
        if index_col is not None:
            df[index_col] = df[index_col].astype(str)
            df = df.set_index(index_col).sort_index()
        self.df: pd.DataFrame = df

        self._ids = list(self.df.index)

    @staticmethod
    def _as_matrix(value):
        try:
            return value.as_matrix()
        except AttributeError:
            return value

    @property
    def ids(self):
        return self._ids

    def get(self, index, col):
        return self._as_matrix(self.df.loc[index, col])

    def get_global_path(self, index, col) -> np.ndarray:
        """
        Join the slice's result with the data frame's `path`.
        Often data frames contain path to data, this is a convenient way to obtain
        the global path.

        Parameters
        ----------
        index,col: same as in pandas.DataFrame.loc

        Returns
        -------
        np.ndarray
        """
        slc = self.df.loc[index, col]
        if type(slc) is str:
            slc = os.path.join(self.path, slc)
        else:
            slc = slc.apply(lambda x: os.path.join(self.path, x))
        return self._as_matrix(slc)

    def load(self, index, col, loader=load_image):
        paths = self.get_global_path(index, col)
        shape = paths.shape
        result = np.array(list(map(loader, paths.flatten())))
        return result.reshape(shape + result[0].shape)
