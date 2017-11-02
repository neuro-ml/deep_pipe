import numpy as np


def get_coordinate_features(shape, center_idx, patch_size):
    assert len(shape) == len(patch_size)

    shape = np.asarray(shape)
    center_idx = np.asarray(center_idx)
    patch_size = np.asarray(patch_size)

    lb = center_idx - patch_size // 2
    rb = lb + patch_size

    spaces = [(np.arange(l, r) + 0.5) / s for l, r, s in zip(lb, rb, shape)]

    return np.stack(reversed(np.meshgrid(*reversed(spaces))))
