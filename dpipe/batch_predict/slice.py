from functools import partial

import numpy as np

from dpipe.medim.slices import iterate_slices
from .base import DivideCombine


class Slice(DivideCombine):
    """Breaks the incoming tensor into slices along the given axis and feeds them into the network."""

    def __init__(self, axis: int = -1):
        super().__init__(partial(iterate_slices, axis=axis), partial(np.stack, axis=axis))
