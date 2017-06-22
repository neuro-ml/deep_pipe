import random

import numpy as np

from ..utils import extract


def extract_patch(x: np.ndarray, center_idx: np.array, spatial_dims: np.array,
                  patch_size: np.array):
    """Function extracts patch of specific spatial size and with specified 
    center from x

    Parameters
    ----------
    x: Array with data. Some of it's dimensions are spatial. We extract 
        spatial patch specified by spatial location and spatial size. If
        available patch is smaller than required, we pad with zeroes.
    center_idx: Location of the center of the patch. Components
        correspond to spatial dimensions. If some of patch size components was
        even, patch center is supposed to be on the right center pixel.
    spatial_dims: Which of x's dimensions consider as spatial. Accepts
        negative parameters.
    patch_size: Spatial patch size. Output will have original shape for
        non-spatial dimensions and patch_size shape for spatial dimensions. 
    """
    start = center_idx - patch_size // 2
    end = start + patch_size

    padding = np.zeros((x.ndim, 2), dtype=int)
    spatial_shape = np.array([extract(o.shape, spatial_dims) for o in x])

    padding[spatial_dims, 0] = np.maximum(-start, 0)
    padding[spatial_dims, 1] = np.maximum(end - spatial_shape, 0)

    start = np.maximum(start, 0)
    end = np.minimum(end, spatial_shape)

    slices = [slice(None)] * x[0].ndim
    for i, s in enumerate(spatial_dims):
        slices[s] = slice(start[i], end[i])

    return np.pad(x[slices], padding, mode='constant')


def uniform(xs: list, patch_sizes: list, *, batch_size: int,
            spatial_dims: list):
    """Patch iterator with uniformed distribution over spatial dimensions.
    First we choose patch for the last patch size and then for the rest."""
    patch_sizes = np.array(patch_sizes)
    assert np.all((patch_sizes % 2)[0:1] == (patch_sizes % 2))

    n_sources = len(xs)
    n_objects = len(xs[0])

    spatial_dims = np.array(spatial_dims)
    spatial_shapes = get_spatial_sizes(xs, spatial_dims)

    # Get max spatial starting index for the last data
    max_spatial_idx = spatial_shapes - patch_sizes[-1] + 1

    # Create batches variables with the right shape and type
    batches = []
    for i in range(n_sources):
        shape = np.array(xs[i][0].shape)
        shape[spatial_dims] = patch_sizes[i]
        batches.append(np.zeros((batch_size, *shape), dtype=xs[i][0].dtype))

    while True:
        # get center index
        idx = np.random.randint(n_objects, size=batch_size)
        start_idx = np.random.rand(batch_size, 3) * max_spatial_idx[idx]
        start_idx = np.int32(np.floor(start_idx))
        center_idx = start_idx + patch_sizes[-1] // 2

        for d in range(n_sources):
            extract_batch_patches(extract(xs[d], idx), batches[d], center_idx,
                                  spatial_dims, patch_sizes[d])

        yield [np.array(b) for b in batches]


def foreground(xs: list, patch_sizes: list, *, batch_size: int, spatial_dims,
               f_fraction: float, f_condition: callable):
    """Patch iterator with uniformed distribution over spatial dimensions for 
    1-f_fraction part of the batch and uniformed distribution over patches with
    f_condition==true in the center for f_fraction part of the batch.
    First we choose patch for the last patch size and then for the rest."""
    patch_sizes = np.array(patch_sizes)
    assert np.all((patch_sizes % 2)[0:1] == (patch_sizes % 2))

    n_sources = len(xs)
    n_objects = len(xs[0])

    f_objects = int(batch_size * f_fraction)

    spatial_dims = np.array(spatial_dims)
    spatial_shapes = get_spatial_sizes(xs, spatial_dims)

    # Get max spatial starting index for the last data
    max_spatial_idx = spatial_shapes - patch_sizes[-1] + 1

    # Get conditional indices
    conditional_indices = []
    for i in range(n_objects):
        c = np.argwhere(f_condition(xs[-1][i]))

        l_bound = c - patch_sizes[-1] // 2
        r_bound = c + patch_sizes[-1] // 2 + patch_sizes[-1]%2

        # Remove centers that are too left
        c = c[np.all((l_bound >= 0) & (r_bound <= spatial_shapes[i]), axis=1)]
        conditional_indices.append(c)

    # Create batches variables with the right shape and type
    batches = []
    for i in range(n_sources):
        shape = np.array(xs[i][0].shape)
        shape[spatial_dims] = patch_sizes[i]
        batches.append(np.zeros((batch_size, *shape), dtype=xs[i][0].dtype))

    while True:
        # get center index
        idx = np.random.randint(n_objects, size=batch_size)
        start_idx = np.random.rand(batch_size, 3) * max_spatial_idx[idx]
        start_idx = np.int32(np.floor(start_idx))
        center_idx = start_idx + patch_sizes[-1] // 2

        for i in range(f_objects):
            center_idx[i] = random.choice(conditional_indices[idx[i]])

        for d in range(n_sources):
            extract_batch_patches(extract(xs[d], idx), batches[d], center_idx,
                                  spatial_dims, patch_sizes[d])

        yield [np.array(b) for b in batches]
