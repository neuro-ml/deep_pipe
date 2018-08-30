import numpy as np

from dpipe.medim.divide import compute_n_parts_per_axis, divide_spatial, combine
from dpipe.medim.shape_utils import compute_shape_from_spatial

from .shape_state import BatchPredictorShapeState
from .utils import pad_spatial_size, slice_spatial_size

spatial_dims = (-3, -2, -1)
zero_spatial_intersection_size = np.zeros(len(spatial_dims), dtype=int)


class Patch3DPredictor(BatchPredictorShapeState):
    """Batch predictor that can be used when need to split 3d image into patches, usually because network can't process
    whole image because of the memory limitations.

    Network is assumed to return `y_patch_size`, given that `x_patch_size` input is provided.

    So, if `x_patch_size` is more than `y_patch_size`, network is assumed to consume
    border voxels and there will be appropriate padding `padding_mode` on the image borders and x patches will
    intersect accordingly.

    If patches on the border are smaller than the provided patch size, it will be padded, so that network will always
    process only `x_patch_size` inputs.
    """

    def __init__(self, x_patch_size: list, y_patch_size: list, padding_mode: str = 'min'):
        self.x_patch_size = np.array(x_patch_size)
        self.y_patch_size = np.array(y_patch_size)

        np.testing.assert_equal(self.x_patch_size % 2, self.y_patch_size % 2)
        np.testing.assert_array_less(self.y_patch_size, self.x_patch_size + 1)

        self.x_spatial_intersection_size = (self.x_patch_size - self.y_patch_size) // 2

        assert padding_mode == 'min'

    def divide_x(self, x):
        return divide_spatial(x, spatial_patch_size=self.x_patch_size, spatial_dims=list(spatial_dims),
                              spatial_intersection_size=self.x_spatial_intersection_size,
                              padding_values=x.min(axis=spatial_dims, keepdims=True))

    def divide_y(self, y):
        return divide_spatial(y, spatial_patch_size=self.y_patch_size, spatial_dims=list(spatial_dims),
                              spatial_intersection_size=zero_spatial_intersection_size)

    def combine_y(self, y_parts, x_shape):
        complete_shape = compute_shape_from_spatial(y_parts[0].shape, x_shape[-3:], spatial_dims=spatial_dims)
        return combine(y_parts, complete_shape)


def find_fixed_spatial_size(spatial_size, spatial_patch_size):
    return compute_n_parts_per_axis(spatial_size, spatial_patch_size) * spatial_patch_size


class Patch3DFixedPredictor(Patch3DPredictor):
    """Batch predictor that can be used when need to split 3d image into patches, usually because network can't process
    whole image because of the memory limitations.

    Network is assumed to return `y_patch_size`, given that `x_patch_size` input is provided.

    So, if `x_patch_size` is more than `y_patch_size`, network is assumed to consume
    border voxels and there will be appropriate padding `padding_mode` on the image borders and x patches will
    intersect accordingly.

    If patches on the border are smaller than the provided patch size, it will be padded, so that network will always
    process only `x_patch_size` inputs.
    """

    def divide_x(self, x):
        spatial_size = np.array(x.shape)[list(spatial_dims)]
        fixed_spatial_size = find_fixed_spatial_size(spatial_size, self.y_patch_size)
        x_padded = pad_spatial_size(x, fixed_spatial_size, spatial_dims)
        return super().divide_x(x_padded)

    def divide_y(self, y):
        spatial_size = np.array(y.shape)[list(spatial_dims)]
        fixed_spatial_size = find_fixed_spatial_size(spatial_size, self.y_patch_size)
        y_padded = pad_spatial_size(y, fixed_spatial_size, spatial_dims)
        return super().divide_y(y_padded)

    def combine_y(self, y_parts, x_shape):
        spatial_size = np.array(x_shape)[list(spatial_dims)]
        fixed_spatial_size = find_fixed_spatial_size(spatial_size, self.y_patch_size)
        y_pred = super().combine_y(y_parts, compute_shape_from_spatial(x_shape, fixed_spatial_size, spatial_dims))
        y_pred = slice_spatial_size(y_pred, spatial_size, spatial_dims)
        return y_pred
