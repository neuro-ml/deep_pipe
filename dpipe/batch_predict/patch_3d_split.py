import numpy as np

import dpipe.medim as medim
from dpipe.config import register

spatial_dims = (-3, -2, -1)
zero_spatial_intersection_size = np.zeros(len(spatial_dims), dtype=int)


def split(x, x_intersection_sizes: np.array, x_patch_sizes: np.array,
          padding_values):
    xs_parts = []
    for x_intersection_size, x_patch_size in zip(x_intersection_sizes,
                                                 x_patch_sizes):
        x_parts = medim.split.divide_spatial(
            x, spatial_patch_size=x_patch_size, spatial_dims=list(spatial_dims),
            spatial_intersection_size=x_intersection_size,
            padding_values=padding_values,
        )
        xs_parts.append([x_part[None, :] for x_part in x_parts])

    assert all([len(x_parts) == len(xs_parts[0]) for x_parts in xs_parts])
    return xs_parts


@register(module_name='patch_3d')
class Patch3DPredictor:
    def __init__(self, x_spatial_patch_sizes: list, y_spatial_patch_size: list,
                 padding_mode: str):
        self.x_spatial_patch_sizes = np.array(x_spatial_patch_sizes)
        self.y_spatial_patch_size = np.array(y_spatial_patch_size)
        assert (np.all(self.x_spatial_patch_sizes % 2 == 1) and
                np.all(self.y_spatial_patch_size % 2 == 1))

        self.x_spatial_intersection_sizes = (self.x_spatial_patch_sizes -
                                             self.y_spatial_patch_size) // 2

        assert padding_mode == 'min'

    def validate(self, x, y, validate_fn):
        xs_batches = split(
            x, self.x_spatial_intersection_sizes, self.x_spatial_patch_sizes,
            x.min(axis=spatial_dims, keepdims=True)
        )
        y_parts_true = medim.split.divide_spatial(
            y, spatial_patch_size=self.y_spatial_patch_size,
            spatial_intersection_size=zero_spatial_intersection_size,
            spatial_dims=list(spatial_dims))
        y_parts_true = [y_part[None, :] for y_part in y_parts_true]

        weights = []
        losses = []
        y_preds = []
        for inputs in zip(*xs_batches, y_parts_true):
            y_pred, loss = validate_fn(*inputs)
            y_preds.append(y_pred)
            losses.append(loss)
            weights.append(y_pred.size)

        loss = np.average(losses, weights=weights)
        complete_shape = medim.shape_utils.compute_shape_from_spatial(
            y_preds[0].shape, x.shape[-3:], spatial_dims=spatial_dims
        )
        return medim.split.combine(y_preds, complete_shape), loss

    def predict(self, x, predict_fn):
        padding_values = x.min(axis=spatial_dims, keepdims=True)
        xs_batches = split(
            x, self.x_spatial_intersection_sizes, self.x_spatial_patch_sizes,
            padding_values
        )
        y_preds = [predict_fn(*inputs) for inputs in zip(*xs_batches)]
        complete_shape = medim.shape_utils.compute_shape_from_spatial(
            y_preds[0].shape, x.shape[-3:], spatial_dims=spatial_dims
        )
        return medim.split.combine(y_preds, complete_shape)
