from warnings import warn

warn(
    'This module is deprecated and will be deleted soon. Use `dpipe.layers` directly.', DeprecationWarning
)  # 04.06.19

from .structure import Lambda
from .shape import PyramidPooling, Reshape, InterpolateToInput
