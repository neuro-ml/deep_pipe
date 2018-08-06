import random

import pdp
import numpy as np

from dpipe.medim import patch
from dpipe.medim.box import get_centered_box
from dpipe.medim.checks import check_len
from .blocks import cache_block_function, make_source_random, make_block_load_x_y, make_batch_blocks

SPATIAL_DIMS = (-3, -2, -1)
DEFAULT_BUFFER_SIZE = 25


def make_block_find_padding():
    @cache_block_function
    def add_padding(o):
        return {'padding': np.min(o['x'], axis=SPATIAL_DIMS, keepdims=True), **o}

    return pdp.One2One(add_padding, buffer_size=DEFAULT_BUFFER_SIZE)


def make_init_blocks(ids, load_x, load_y):
    return (make_source_random(ids), make_block_load_x_y(load_x, load_y, buffer_size=len(ids)),
            make_block_find_padding())


def sample_center_uniformly(shape, patch_size, spatial_dims):
    spatial_shape = np.array(shape)[list(spatial_dims)]
    if np.all(patch_size <= spatial_shape):
        return patch.sample_box_center_uniformly(shape=spatial_shape, box_size=patch_size)
    else:
        return spatial_shape // 2


def make_block_extract_patches(x_patch_size, y_patch_size, spatial_dims):
    x_patch_size = np.array(x_patch_size)
    y_patch_size = np.array(y_patch_size)

    def extract_patches(o):
        x_spatial_box = get_centered_box(o['center'], x_patch_size)
        y_spatial_box = get_centered_box(o['center'], y_patch_size)
        x = patch.extract_patch_spatial_box(o['x'], spatial_box=x_spatial_box, spatial_dims=spatial_dims,
                                            padding_values=o['padding'])
        y = patch.extract_patch_spatial_box(o['y'], spatial_box=y_spatial_box, spatial_dims=spatial_dims,
                                            padding_values=0)
        return x, y

    return pdp.One2One(extract_patches, buffer_size=DEFAULT_BUFFER_SIZE)


def make_block_sample_uniformly(box_size, spatial_dims):
    box_size = np.array(box_size)

    def sample(o):
        return {**o, 'center': sample_center_uniformly(o['y'].shape, box_size, spatial_dims)}

    return pdp.One2One(sample, buffer_size=DEFAULT_BUFFER_SIZE)


def make_patch_3d_iter(ids, load_x, load_y, *, batch_size, x_patch_size, y_patch_size, buffer_size=10):
    check_len(x_patch_size, y_patch_size)

    return pdp.Pipeline(*make_init_blocks(ids, load_x, load_y),
                        make_block_sample_uniformly(y_patch_size, spatial_dims=SPATIAL_DIMS),
                        make_block_extract_patches(x_patch_size, y_patch_size, spatial_dims=SPATIAL_DIMS),
                        *make_batch_blocks(batch_size, buffer_size=buffer_size))


def find_nonzero_3d(y):
    if len(y.shape) == 3:
        mask = y > 0
    elif len(y.shape) == 4:
        mask = np.any(y, axis=0)
    else:
        raise ValueError('wrong number of dimensions ')
    return mask


def make_block_find_nonzero():
    @cache_block_function
    def add_cancer(o):
        return {**o, 'nonzero': np.argwhere(find_nonzero_3d(o['y']))}

    return pdp.One2One(add_cancer, buffer_size=DEFAULT_BUFFER_SIZE)


def make_block_sample_stratified(nonzero_fraction, spatial_box_size, spatial_dims):
    spatial_box_size = np.array(spatial_box_size)

    def sample_patch_center_strat(o):
        if len(o['nonzero']) > 0 and np.random.uniform() < nonzero_fraction:
            center = tuple(random.choice(o['nonzero']))
        else:
            center = sample_center_uniformly(o['y'].shape, spatial_box_size, spatial_dims=spatial_dims)
        return {**o, 'center': center}

    return pdp.One2One(sample_patch_center_strat, buffer_size=DEFAULT_BUFFER_SIZE)


def make_patch_3d_strat_iter(ids, load_x, load_y, *, batch_size, x_patch_size, y_patch_size,
                             nonzero_fraction, buffer_size=10):
    check_len(x_patch_size, y_patch_size)
    y_patch_size = np.array(y_patch_size)
    return pdp.Pipeline(*make_init_blocks(ids, load_x, load_y),
                        make_block_find_nonzero(),
                        make_block_sample_stratified(nonzero_fraction, y_patch_size, spatial_dims=SPATIAL_DIMS),
                        make_block_extract_patches(x_patch_size, y_patch_size, spatial_dims=SPATIAL_DIMS),
                        *make_batch_blocks(batch_size, buffer_size=buffer_size))
