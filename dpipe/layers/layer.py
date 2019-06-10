from warnings import warn

warn(
    'This module is deprecated and will be deleted soon. Use `dpipe.layers.structure` instead.', DeprecationWarning
)  # 04.06.19

from .structure import PyramidPooling, Lambda, Reshape, InterpolateToInput
