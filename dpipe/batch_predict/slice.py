import numpy as np

from dpipe.batch_predict.simple import add_dims
from dpipe.medim.slices import iterate_slices
from .base import DivideCombine


class Slice(DivideCombine):
    """Breaks the incoming tensor into slices along the given axis and feeds them into the network."""

    def __init__(self, axis: int = -1):
        super().__init__(lambda *x: iterate_slices(*add_dims(*x), axis=axis), lambda x: np.stack(x, axis=axis)[0])
