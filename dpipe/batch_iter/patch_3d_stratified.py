from functools import lru_cache, partial
import random

import numpy as np
import pdp

from dpipe.medim import patch
from dpipe.medim.augmentation import spatial_augmentation_strict, random_flip
from dpipe.config import register


class Patient:
    def __init__(self, patient_id, x, y):
        self.patient_id = patient_id
        self.x, self.y = x, y
        assert all(np.array(self.x.shape[1:]) == np.array(self.y.shape[-3:])), \
            (f"Wrong shape was provided for patient {patient_id}\n"
             f"x.shape = {self.x.shape} y.shape = {self.y.shape}")

    def __hash__(self):
        return hash(self.patient_id)


def find_cancer_and_padding_values(x, y, y_patch_size, spatial_dims):
    if len(y.shape) == 3:
        mask = y > 0
    elif len(y.shape) == 4:
        mask = np.any(y, axis=0)
    else:
        raise ValueError('wrong number of dimensions ')
    assert np.any(mask), f'No cancer voxels found'
    cancer_center_indices = patch.find_masked_patch_center_indices(mask, patch_size=y_patch_size)
    padding_values = np.min(x, axis=tuple(spatial_dims), keepdims=True)

    return x, y, cancer_center_indices, padding_values


def extract_big_patches(x, y, cancer_center_indices, padding_values, nonzero_fraction,
                        big_x_patch_size, y_patch_size, spatial_dims):
    if np.random.uniform() < nonzero_fraction:
        spatial_center_idx = random.choice(cancer_center_indices)
    else:
        spatial_center_idx = patch.sample_uniform_center_index(
            x_shape=np.array(x.shape), spatial_patch_size=y_patch_size, spatial_dims=spatial_dims
        )
    x = patch.extract_patch(x, spatial_center_idx=spatial_center_idx, spatial_dims=spatial_dims,
                            spatial_patch_size=big_x_patch_size, padding_values=padding_values)
    y = patch.extract_patch(y, spatial_center_idx=spatial_center_idx, spatial_dims=spatial_dims,
                            spatial_patch_size=y_patch_size, padding_values=0)
    return x, y


def extract_patches(x, y, big_x_patch_center_idx, x_patch_sizes, spatial_dims):
    xs = [patch.extract_patch(x, spatial_center_idx=big_x_patch_center_idx, spatial_dims=spatial_dims,
                              spatial_patch_size=patch_size) for patch_size in x_patch_sizes]
    return (*xs, y)


@register('patch_3d_strat')
def make_patch_3d_strat_iter(ids, load_x, load_y, *, batch_size, x_patch_sizes, y_patch_size, nonzero_fraction,
                             buffer_size):
    x_patch_sizes = np.array(x_patch_sizes)
    y_patch_size = np.array(y_patch_size)

    assert np.all(x_patch_sizes % 2 == 1) and np.all(y_patch_size % 2 == 1)

    spatial_dims = [-3, -2, -1]

    random_seq = iter(partial(random.choice, ids), None)

    @lru_cache(len(ids))
    def _load_patient(patient_id):
        return Patient(patient_id, load_x(patient_id), load_y(patient_id))

    @lru_cache(len(ids))
    def _find_cancer_and_padding_values(patient: Patient):
        return find_cancer_and_padding_values(patient.x, patient.y, y_patch_size=y_patch_size,
                                              spatial_dims=spatial_dims)

    big_x_patch_size = np.max(x_patch_sizes, axis=0)
    big_x_patch_center_idx = big_x_patch_size // 2

    @pdp.pack_args
    def _extract_big_patches(x, y, cancer_center_indices, padding_values):
        return extract_big_patches(x, y, cancer_center_indices=cancer_center_indices, padding_values=padding_values,
                                   nonzero_fraction=nonzero_fraction, big_x_patch_size=big_x_patch_size,
                                   y_patch_size=y_patch_size, spatial_dims=spatial_dims)

    @pdp.pack_args
    def _extract_patches(x, y):
        return extract_patches(x, y, big_x_patch_center_idx=big_x_patch_center_idx, x_patch_sizes=x_patch_sizes,
                               spatial_dims=spatial_dims)

    return pdp.Pipeline(
        pdp.Source(random_seq, buffer_size=3),
        pdp.One2One(_load_patient, buffer_size=len(ids)),
        pdp.One2One(_find_cancer_and_padding_values, buffer_size=len(ids)),
        pdp.One2One(_extract_big_patches, buffer_size=batch_size),
        pdp.One2One(_extract_patches, buffer_size=batch_size),
        pdp.Many2One(chunk_size=batch_size, buffer_size=3),
        pdp.One2One(pdp.combine_batches, buffer_size=buffer_size)
    )


@register('patch_3d_strat_quantiles')
def make_patch_3d_strat_iter_quantiles(ids, load_x, load_y, *, batch_size, x_patch_sizes, y_patch_size,
                                       nonzero_fraction, buffer_size, n_quantiles):
    x_patch_sizes = np.array(x_patch_sizes)
    y_patch_size = np.array(y_patch_size)

    assert np.all(x_patch_sizes % 2 == 1) and np.all(y_patch_size % 2 == 1)

    spatial_dims = [-3, -2, -1]

    random_seq = iter(partial(random.choice, ids), None)

    @lru_cache(len(ids))
    def _load_patient(patient_id):
        return Patient(patient_id, load_x(patient_id), load_y(patient_id))

    @lru_cache(len(ids))
    def _find_cancer_and_padding_values_and_quantiles_(patient: Patient):
        x, y, cancer_ids, padding_vals = find_cancer_and_padding_values(
            patient.x, patient.y, y_patch_size=y_patch_size, spatial_dims=spatial_dims
        )
        quantiles = np.percentile(x, np.linspace(0, 100, n_quantiles))
        return x, y, cancer_ids, padding_vals, quantiles

    big_x_patch_size = np.max(x_patch_sizes, axis=0)
    big_x_patch_center_idx = big_x_patch_size // 2

    @pdp.pack_args
    def _extract_big_patches(x, y, cancer_center_indices, padding_values, quantiles):
        x, y = extract_big_patches(x, y, cancer_center_indices=cancer_center_indices, padding_values=padding_values,
                                   nonzero_fraction=nonzero_fraction, big_x_patch_size=big_x_patch_size,
                                   y_patch_size=y_patch_size, spatial_dims=spatial_dims)
        return x, y, quantiles

    @pdp.pack_args
    def _extract_patches(x, y, quantiles):
        *xs, y = extract_patches(x, y, big_x_patch_center_idx=big_x_patch_center_idx, x_patch_sizes=x_patch_sizes,
                                 spatial_dims=spatial_dims)
        return (*xs, quantiles, y)

    return pdp.Pipeline(
        pdp.Source(random_seq, buffer_size=3),
        pdp.One2One(_load_patient, buffer_size=len(ids)),
        pdp.One2One(_find_cancer_and_padding_values_and_quantiles_, buffer_size=len(ids)),
        pdp.One2One(_extract_big_patches, buffer_size=batch_size),
        pdp.One2One(_extract_patches, buffer_size=batch_size),
        pdp.Many2One(chunk_size=batch_size, buffer_size=3),
        pdp.One2One(pdp.combine_batches, buffer_size=buffer_size)
    )


class ExpirationPool:
    def __init__(self, expiration_time, pool_size):
        self.pool_size = pool_size
        self.expiration_time = expiration_time

        self.data = []
        self.expiration_timer = []

    def is_full(self):
        return len(self.data) == self.pool_size

    def put(self, value):
        assert not self.is_full()
        self.data.append(value)
        self.expiration_timer.append(self.expiration_time)

    def draw(self):
        assert self.is_full()
        i = random.randint(0, self.pool_size - 1)
        value = self.data[i]
        self.expiration_timer[i] -= 1

        assert self.expiration_timer[i] >= 0
        if self.expiration_timer[i] == 0:
            del self.data[i]
            del self.expiration_timer[i]

        return value


@register('patch_3d_strat_augm')
def make_3d_patch_strat_augm_iter(ids, load_x, load_y, *, batch_size, x_patch_sizes, y_patch_size, nonzero_fraction,
                                  buffer_size, expiration_time, pool_size, n_workers):
    x_patch_sizes = np.array(x_patch_sizes)
    y_patch_size = np.array(y_patch_size)

    assert np.all(x_patch_sizes % 2 == 1) and np.all(y_patch_size % 2 == 1)

    spatial_dims = [-3, -2, -1]

    random_seq = iter(partial(random.choice, ids), None)

    @lru_cache(len(ids))
    def _load_patient(patient_id):
        return load_x(patient_id), load_y(patient_id)

    @pdp.pack_args
    def _augment(x, y):
        convert = y.ndim == 3
        if convert:
            unique = np.unique(y)
            y = np.array([y == i for i in unique], dtype=np.float32)

        x, y = spatial_augmentation_strict(x, y, axes=[-3, -2, -1])
        x, y = random_flip(x, y, axes=[-3])

        if convert:
            y = np.argmax(y, axis=0)
            #     restoring old int tensor
            if set(unique) - set(range(len(unique))):
                for i, val in enumerate(unique):
                    y[y == i] = val
        else:
            y = y > 0.5

        return x, y

    @pdp.pack_args
    def _find_cancer_and_padding_values(x, y):
        return find_cancer_and_padding_values(x, y, y_patch_size=y_patch_size, spatial_dims=spatial_dims)

    pool = ExpirationPool(expiration_time=expiration_time, pool_size=pool_size)

    @pdp.pack_args
    def _augmented_pool_sampling(x, y, cancer_center_indices, padding_values):
        nonlocal pool
        pool.put((x, y, cancer_center_indices, padding_values))
        while pool.is_full():
            yield pool.draw()

    big_x_patch_size = np.max(x_patch_sizes, axis=0)
    big_x_patch_center_idx = big_x_patch_size // 2

    @pdp.pack_args
    def _extract_big_patches(x, y, cancer_center_indices, padding_values):
        return extract_big_patches(x, y, cancer_center_indices=cancer_center_indices, padding_values=padding_values,
                                   nonzero_fraction=nonzero_fraction, big_x_patch_size=big_x_patch_size,
                                   y_patch_size=y_patch_size, spatial_dims=spatial_dims)

    @pdp.pack_args
    def _extract_patches(x, y):
        return extract_patches(x, y, big_x_patch_center_idx=big_x_patch_center_idx, x_patch_sizes=x_patch_sizes,
                               spatial_dims=spatial_dims)

    return pdp.Pipeline(
        pdp.Source(random_seq, buffer_size=3),
        pdp.One2One(_load_patient, buffer_size=len(ids)),
        pdp.One2One(_augment, n_workers=n_workers, buffer_size=3),
        pdp.One2One(_find_cancer_and_padding_values, buffer_size=pool_size // 2),
        pdp.One2Many(_augmented_pool_sampling, buffer_size=batch_size),
        pdp.One2One(_extract_big_patches, buffer_size=batch_size),
        pdp.One2One(_extract_patches, buffer_size=batch_size),
        pdp.Many2One(chunk_size=batch_size, buffer_size=3),
        pdp.One2One(pdp.combine_batches, buffer_size=buffer_size)
    )
