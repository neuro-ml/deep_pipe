import numpy as np

from dpipe.config import register
from dpipe.medim.divide import compute_n_parts_per_axis
from .patch_3d import Patch3DPredictor


spatial_dims = (-3, -2, -1)
zero_spatial_intersection_size = np.zeros(len(spatial_dims), dtype=int)


@register(module_name='patch_3d_fixed')
class Patch3DFixedPredictor(Patch3DPredictor):
    def validate(self, x, y, *, validate_fn):
        original_spatial_shape = x.shape[spatial_dims]
        fixed_spatial_shape = compute_n_parts_per_axis(x.shape, 12)
        super().validate(x, y, validate_fn=validate_fn)

    def predict(self, x, *, predict_fn):
        super().predict(x, predict_fn=predict_fn)
