import numpy as np

from dpipe.layers import identity
from dpipe.im import min_max_scale, normalize
from dpipe.im.utils import apply_along_axes, get_mask_volume

almost_eq = np.testing.assert_array_almost_equal


def test_apply():
    x = np.random.rand(3, 10, 10) * 2 + 3
    almost_eq(
        apply_along_axes(normalize, x, axes=(1, 2), percentiles=20),
        normalize(x, percentiles=20, axis=0)
    )

    axes = (0, 2)
    y = apply_along_axes(min_max_scale, x, axes)
    almost_eq(y.max(axes), 1)
    almost_eq(y.min(axes), 0)

    almost_eq(apply_along_axes(identity, x, 1), x)
    almost_eq(apply_along_axes(identity, x, -1), x)
    almost_eq(apply_along_axes(identity, x, (0, 1)), x)
    almost_eq(apply_along_axes(identity, x, (0, 2)), x)


def test_mask_volume():
    mask = np.random.rand(5, 5, 5) >= 0.5
    vol = mask.sum()
    ones = np.ones(5)
    rng = ones.cumsum()

    assert (get_mask_volume(mask, 1, 1, 1) ==
            get_mask_volume(mask, 1, ones, 1) ==
            get_mask_volume(mask, 1, rng, 1, location=True) ==
            get_mask_volume(mask, ones, ones, ones) == vol)

    assert (get_mask_volume(mask, 1, 2, 3) ==
            get_mask_volume(mask, rng * 1, 2, rng * 3, location=True) ==
            get_mask_volume(mask, ones * 1, 2, ones * 3) == vol * 6)
