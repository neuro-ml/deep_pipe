import numpy as np

from dpipe.batch_predict.simple import add_dims, extract_dims
from dpipe.medim.slices import iterate_slices
from .base import DivideCombine


class Slice(DivideCombine):
    """
    Breaks the incoming tensor into slices along the given ``axis`` and feeds them into the network.

    Parameters
    ----------
    axis: int, optional
        the axis along which the slices will be taken from.
    ndims: int, optional
        the number of leading singleton dimensions to add.
    """

    def __init__(self, axis: int = -1, ndims=1):
        super().__init__(lambda *x: iterate_slices(*add_dims(*x, ndims=ndims), axis=axis),
                         lambda x: extract_dims(np.stack(x, axis=axis), ndims))
