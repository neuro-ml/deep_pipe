import random

import pdp
from pdp import product_generator
import numpy as np

from dpipe.medim import patch
from .blocks import cache_function, make_source_random, make_block_load_x_y, make_batch_blocks

spatial_dims = (-3, -2, -1)


def find_cancer(y, y_patch_size):
    if len(y.shape) == 3:
        mask = y > 0
    elif len(y.shape) == 4:
        mask = np.any(y, axis=0)
    else:
        raise ValueError('wrong number of dimensions ')
    assert np.any(mask), f'No cancer voxels found'
    return patch.find_masked_patch_center_indices(mask, patch_size=y_patch_size)


def make_block_find_cancer(y_patch_size, *, buffer_size):
    @cache_function
    def add_cancer(o):
        o['cancer'] = find_cancer(o['y'], y_patch_size)
        return o

    return pdp.One2One(add_cancer, buffer_size=buffer_size)


def make_block_find_padding(buffer_size):
    @cache_function
    def add_padding(o):
        o['padding'] = np.min(o['x'], axis=spatial_dims, keepdims=True)
        return o

    return pdp.One2One(add_padding, buffer_size=buffer_size)


def get_random_center_idx(y, y_patch_size, spatial_dims):
    y_shape = np.array(y.shape)
    if np.all(y_patch_size <= y_shape[list(spatial_dims)]):
        spatial_center_idx = patch.sample_uniform_center_index(x_shape=y_shape, spatial_patch_size=y_patch_size,
                                                               spatial_dims=spatial_dims)
    else:
        spatial_center_idx = y_shape[list(spatial_dims)] // 2
    return spatial_center_idx


def extract_patches(x, patch_sizes, center_idx, padding_values, spatial_dims):
    return [patch.extract_patch(x, spatial_center_idx=center_idx, spatial_dims=spatial_dims,
                                spatial_patch_size=x_patch_size, padding_values=padding_values)
            for x_patch_size in patch_sizes]


def make_patch_3d_iter(ids, load_x, load_y, *, batch_size, x_patch_sizes, y_patch_size, buffer_size):
    x_patch_sizes = np.array(x_patch_sizes)
    y_patch_size = np.array(y_patch_size)

    def _extract_patches(o):
        center_idx = get_random_center_idx(o['y'], y_patch_size, spatial_dims=spatial_dims)

        xs = extract_patches(o['x'], patch_sizes=x_patch_sizes, center_idx=center_idx, padding_values=o['padding'],
                             spatial_dims=spatial_dims)
        y, = extract_patches(o['y'], patch_sizes=[y_patch_size], center_idx=center_idx, padding_values=0,
                             spatial_dims=spatial_dims)

        return (*xs, y)

    return product_generator(
        make_source_random(ids),
        make_block_load_x_y(load_x, load_y, buffer_size=len(ids)),
        make_block_find_padding(buffer_size=len(ids)),
        pdp.One2One(_extract_patches, buffer_size=batch_size),
        *make_batch_blocks(batch_size, buffer_size=buffer_size)
    )


def make_patch_3d_strat_iter(ids, load_x, load_y, *, batch_size, x_patch_sizes, y_patch_size, nonzero_fraction,
                             buffer_size):
    x_patch_sizes = np.array(x_patch_sizes)
    y_patch_size = np.array(y_patch_size)

    def _extract_patches(o):
        if np.random.uniform() < nonzero_fraction:
            center_idx = random.choice(o['cancer'])
        else:
            center_idx = get_random_center_idx(o['y'], y_patch_size, spatial_dims=spatial_dims)

        xs = extract_patches(o['x'], patch_sizes=x_patch_sizes, center_idx=center_idx, padding_values=o['padding'],
                             spatial_dims=spatial_dims)

        y, = extract_patches(o['y'], patch_sizes=[y_patch_size], center_idx=center_idx, padding_values=0,
                             spatial_dims=spatial_dims)
        return (*xs, y)

    return product_generator(
        make_source_random(ids),
        make_block_load_x_y(load_x, load_y, buffer_size=len(ids)),
        make_block_find_padding(len(ids)),
        make_block_find_cancer(y_patch_size, buffer_size=len(ids)),
        pdp.One2One(_extract_patches, buffer_size=batch_size),
        *make_batch_blocks(batch_size, buffer_size=buffer_size)
    )

