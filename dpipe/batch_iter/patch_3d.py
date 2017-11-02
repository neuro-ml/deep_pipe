import random
import functools
import collections

import pdp
import numpy as np

from dpipe.medim import patch
from dpipe.medim.augmentation import spatial_augmentation_strict, random_flip
from dpipe.medim.features import get_coordinate_features
from dpipe.config import register


class Patient(collections.namedtuple('Patient_data', ['patient_id', 'x', 'y'])):
    def __hash__(self):
        return hash(self.patient_id)


def find_cancer(y, y_patch_size):
    if len(y.shape) == 3:
        mask = y > 0
    elif len(y.shape) == 4:
        mask = np.any(y, axis=0)
    else:
        raise ValueError('wrong number of dimensions ')
    assert np.any(mask), f'No cancer voxels found'
    return patch.find_masked_patch_center_indices(mask, patch_size=y_patch_size)


def get_random_center_idx(y, y_patch_size, spatial_dims):
    y_shape = np.array(y.shape)
    if np.all(y_patch_size <= y_shape[spatial_dims]):
        spatial_center_idx = patch.sample_uniform_center_index(x_shape=y_shape, spatial_patch_size=y_patch_size,
                                                               spatial_dims=spatial_dims)
    else:
        spatial_center_idx = y_shape[spatial_dims] // 2
    return spatial_center_idx


def extract_patches(x, patch_sizes, center_idx, padding_values, spatial_dims):
    return [patch.extract_patch(x, spatial_center_idx=center_idx, spatial_dims=spatial_dims,
                                spatial_patch_size=x_patch_size, padding_values=padding_values)
            for x_patch_size in patch_sizes]


@register('patch_3d')
def make_patch_3d_iter(ids, load_x, load_y, *, batch_size, x_patch_sizes, y_patch_size, buffer_size):
    x_patch_sizes = np.array(x_patch_sizes)
    y_patch_size = np.array(y_patch_size)

    assert np.all(x_patch_sizes % 2 == 1) and np.all(y_patch_size % 2 == 1)

    spatial_dims = [-3, -2, -1]

    random_seq = iter(functools.partial(random.choice, ids), None)

    @functools.lru_cache(len(ids))
    def _load_patient(patient_id):
        return Patient(patient_id, load_x(patient_id), load_y(patient_id))

    @functools.lru_cache(len(ids))
    def _find_padding_values(patient: Patient):
        return patient.x, patient.y, np.min(patient.x, axis=tuple(spatial_dims), keepdims=True)

    @pdp.pack_args
    def _extract_patches(x, y, padding_values):
        center_idx = get_random_center_idx(y, y_patch_size, spatial_dims=spatial_dims)

        xs = extract_patches(x, patch_sizes=x_patch_sizes, center_idx=center_idx, padding_values=padding_values,
                             spatial_dims=spatial_dims)
        y, = extract_patches(y, patch_sizes=[y_patch_size], center_idx=center_idx, padding_values=0,
                            spatial_dims=spatial_dims)

        return (*xs, y)

    return pdp.Pipeline(
        pdp.Source(random_seq, buffer_size=3),
        pdp.One2One(_load_patient, buffer_size=len(ids)),
        pdp.One2One(_find_padding_values, buffer_size=len(ids)),
        pdp.One2One(_extract_patches, buffer_size=batch_size),
        pdp.Many2One(chunk_size=batch_size, buffer_size=3),
        pdp.One2One(pdp.combine_batches, buffer_size=buffer_size)
    )


def find_cancer_and_padding_values(x, y, y_patch_size, spatial_dims):
    return x, y, find_cancer(y, y_patch_size), np.min(x, axis=tuple(spatial_dims), keepdims=True)


@register('patch_3d_strat')
def make_patch_3d_strat_iter(ids, load_x, load_y, *, batch_size, x_patch_sizes, y_patch_size, nonzero_fraction,
                             buffer_size):
    x_patch_sizes = np.array(x_patch_sizes)
    y_patch_size = np.array(y_patch_size)

    assert np.all(x_patch_sizes % 2 == 1) and np.all(y_patch_size % 2 == 1)

    spatial_dims = [-3, -2, -1]

    random_seq = iter(functools.partial(random.choice, ids), None)

    @functools.lru_cache(len(ids))
    def _load_patient(patient_id):
        return Patient(patient_id, load_x(patient_id), load_y(patient_id))

    @functools.lru_cache(len(ids))
    def _find_cancer_and_padding_values(patient: Patient):
        return find_cancer_and_padding_values(patient.x, patient.y, y_patch_size=y_patch_size,
                                              spatial_dims=spatial_dims)

    @pdp.pack_args
    def _extract_patches(x, y, cancer_center_indices, padding_values):
        if np.random.uniform() < nonzero_fraction:
            center_idx = random.choice(cancer_center_indices)
        else:
            center_idx = get_random_center_idx(y, y_patch_size, spatial_dims=spatial_dims)

        xs = extract_patches(x, patch_sizes=x_patch_sizes, center_idx=center_idx, padding_values=padding_values,
                             spatial_dims=spatial_dims)
        y, = extract_patches(y, patch_sizes=[y_patch_size], center_idx=center_idx, padding_values=0,
                            spatial_dims=spatial_dims)
        return (*xs, y)


    return pdp.Pipeline(
        pdp.Source(random_seq, buffer_size=3),
        pdp.One2One(_load_patient, buffer_size=len(ids)),
        pdp.One2One(_find_cancer_and_padding_values, buffer_size=len(ids)),
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

    random_seq = iter(functools.partial(random.choice, ids), None)

    @functools.lru_cache(len(ids))
    def _load_patient(patient_id):
        return Patient(patient_id, load_x(patient_id), load_y(patient_id))

    @functools.lru_cache(len(ids))
    def _find_cancer_and_padding_values_and_quantiles_(patient: Patient):
        x, y, cancer_ids, padding_vals = find_cancer_and_padding_values(
            patient.x, patient.y, y_patch_size=y_patch_size, spatial_dims=spatial_dims
        )
        quantiles = np.percentile(x, np.linspace(0, 100, n_quantiles))
        return x, y, cancer_ids, padding_vals, quantiles

    @pdp.pack_args
    def _extract_patches(x, y, cancer_center_indices, padding_values, quantiles):
        if np.random.uniform() < nonzero_fraction:
            center_idx = random.choice(cancer_center_indices)
        else:
            center_idx = get_random_center_idx(y, y_patch_size, spatial_dims=spatial_dims)

        xs = extract_patches(x, patch_sizes=x_patch_sizes, center_idx=center_idx, padding_values=padding_values,
                             spatial_dims=spatial_dims)
        y, = extract_patches(y, patch_sizes=[y_patch_size], center_idx=center_idx, padding_values=0,
                            spatial_dims=spatial_dims)
        return (*xs, quantiles, y)

    return pdp.Pipeline(
        pdp.Source(random_seq, buffer_size=3),
        pdp.One2One(_load_patient, buffer_size=len(ids)),
        pdp.One2One(_find_cancer_and_padding_values_and_quantiles_, buffer_size=len(ids)),
        pdp.One2One(_extract_patches, buffer_size=batch_size),
        pdp.Many2One(chunk_size=batch_size, buffer_size=3),
        pdp.One2One(pdp.combine_batches, buffer_size=buffer_size)
    )


@register('patch_3d_strat_quantiles_coordinates')
def make_patch_3d_strat_iter_quantiles(ids, load_x, load_y, *, batch_size, x_patch_sizes, y_patch_size,
                                       nonzero_fraction, buffer_size, n_quantiles):
    x_patch_sizes = np.array(x_patch_sizes)
    y_patch_size = np.array(y_patch_size)

    assert np.all(x_patch_sizes % 2 == 1) and np.all(y_patch_size % 2 == 1)

    spatial_dims = [-3, -2, -1]

    random_seq = iter(functools.partial(random.choice, ids), None)

    @functools.lru_cache(len(ids))
    def _load_patient(patient_id):
        return Patient(patient_id, load_x(patient_id), load_y(patient_id))

    @functools.lru_cache(len(ids))
    def _find_cancer_and_padding_values_and_quantiles_(patient: Patient):
        x, y, cancer_ids, padding_vals = find_cancer_and_padding_values(
            patient.x, patient.y, y_patch_size=y_patch_size, spatial_dims=spatial_dims
        )
        quantiles = np.percentile(x, np.linspace(0, 100, n_quantiles))
        return x, y, cancer_ids, padding_vals, quantiles

    @pdp.pack_args
    def _extract_patches(x, y, cancer_center_indices, padding_values, quantiles):
        if np.random.uniform() < nonzero_fraction:
            center_idx = random.choice(cancer_center_indices)
        else:
            center_idx = get_random_center_idx(y, y_patch_size, spatial_dims=spatial_dims)

        xs = extract_patches(x, patch_sizes=x_patch_sizes, center_idx=center_idx, padding_values=padding_values,
                             spatial_dims=spatial_dims)
        y, = extract_patches(y, patch_sizes=[y_patch_size], center_idx=center_idx, padding_values=0,
                            spatial_dims=spatial_dims)

        center_patch = get_coordinate_features(np.array(x.shape)[spatial_dims], center_idx, y_patch_size)

        return (*xs, center_patch, quantiles, y)

    return pdp.Pipeline(
        pdp.Source(random_seq, buffer_size=3),
        pdp.One2One(_load_patient, buffer_size=len(ids)),
        pdp.One2One(_find_cancer_and_padding_values_and_quantiles_, buffer_size=len(ids)),
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

    random_seq = iter(functools.partial(random.choice, ids), None)

    @functools.lru_cache(len(ids))
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

    @pdp.pack_args
    def _extract_patches(x, y, cancer_center_indices, padding_values):
        if np.random.uniform() < nonzero_fraction:
            center_idx = random.choice(cancer_center_indices)
        else:
            center_idx = get_random_center_idx(y, y_patch_size, spatial_dims=spatial_dims)

        xs = extract_patches(x, patch_sizes=x_patch_sizes, center_idx=center_idx, padding_values=padding_values,
                             spatial_dims=spatial_dims)
        y, = extract_patches(y, patch_sizes=[y_patch_size], center_idx=center_idx, padding_values=0,
                            spatial_dims=spatial_dims)
        return (*xs, y)

    return pdp.Pipeline(
        pdp.Source(random_seq, buffer_size=3),
        pdp.One2One(_load_patient, buffer_size=len(ids)),
        pdp.One2One(_augment, n_workers=n_workers, buffer_size=3),
        pdp.One2One(_find_cancer_and_padding_values, buffer_size=pool_size // 2),
        pdp.One2Many(_augmented_pool_sampling, buffer_size=batch_size),
        pdp.One2One(_extract_patches, buffer_size=batch_size),
        pdp.Many2One(chunk_size=batch_size, buffer_size=3),
        pdp.One2One(pdp.combine_batches, buffer_size=buffer_size)
    )
