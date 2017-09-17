from functools import lru_cache, partial
from random import choice

import numpy as np

from dpipe.medim import patch
import dpipe.externals.pdp.pdp as pdp
from dpipe.config import register


class Patient:
    def __init__(self, patient_id, x, y):
        self.patient_id = patient_id
        self.x, self.y = x, y
        assert (all(np.array(self.x.shape[1:]) == np.array(self.y.shape[-3:])),
                f"Wrong shape was provided for patient {patient_id}\n"
                f"x.shape = {self.x.shape} y.shape = {self.y.shape}")

    def __hash__(self):
        return hash(self.patient_id)


def pad_channel_min(x, padding):
    shape = np.array(x.shape)
    y = np.zeros([shape[0]] + list(2 * padding + shape[1:]))
    y[:] = np.min(x, axis=(1, 2, 3), keepdims=True)
    y[[slice(None)] + [*map(slice, padding, -padding)]] = x
    return y


@register('3d_patch_strat')
def make_3d_patch_stratified_iter(
        ids, load_x, load_y, *, batch_size, x_patch_sizes,
        y_patch_size, nonzero_fraction, buffer_size=10):
    x_patch_sizes = np.array(x_patch_sizes)
    y_patch_size = np.array(y_patch_size)

    assert np.all(x_patch_sizes % 2 == 1) and np.all(y_patch_size % 2 == 1)

    spatial_dims = [-3, -2, -1]

    random_seq = iter(partial(choice, ids), None)

    @lru_cache(maxsize=len(ids))
    def load_patient(name):
        return Patient(name, load_x(name), load_y(name))

    @lru_cache(maxsize=len(ids))
    def find_cancer_and_padding_values(patient: Patient):
        if len(patient.y.shape) == 3:
            mask = patient.y > 0
        elif len(patient.y.shape) == 4:
            mask = np.any(patient.y, axis=0)
        else:
            raise ValueError('wrong number of dimensions ')
        cancer_center_indices = patch.find_masked_patch_center_indices(
            mask, patch_size=y_patch_size
        )

        padding_values = np.min(patient.x, axis=tuple(spatial_dims),
                                keepdims=True)

        return patient.x, patient.y, cancer_center_indices, padding_values

    big_x_patch_size = np.max(x_patch_sizes, axis=0)
    big_x_patch_center_idx = big_x_patch_size // 2

    @pdp.pack_args
    def extract_big_patches(x, y, cancer_center_indices, padding_values):
        if np.random.uniform() < nonzero_fraction:
            spatial_center_idx = choice(cancer_center_indices)
        else:
            spatial_center_idx = patch.sample_uniform_center_index(
                x_shape=np.array(x.shape), spatial_patch_size=y_patch_size,
                spatial_dims=spatial_dims)

        x = patch.extract_patch(
                  x, spatial_center_idx=spatial_center_idx, spatial_dims=spatial_dims,
                  spatial_patch_size=big_x_patch_size, padding_values=padding_values
        )

        y = patch.extract_patch(
            y, spatial_center_idx=spatial_center_idx, spatial_dims=spatial_dims,
            spatial_patch_size=y_patch_size, padding_values=0
        )

        return x, y

    @pdp.pack_args
    def extract_patches(x, y):
        xs = [patch.extract_patch(
            x, spatial_center_idx=big_x_patch_center_idx,
            spatial_dims=spatial_dims, spatial_patch_size=patch_size
              )
              for patch_size in x_patch_sizes]

        return (*xs, y)

    return pdp.Pipeline(
        pdp.Source(random_seq, buffer_size=3),
        pdp.LambdaTransformer(load_patient, buffer_size=100),
        pdp.LambdaTransformer(find_cancer_and_padding_values, buffer_size=100),
        pdp.LambdaTransformer(extract_big_patches, buffer_size=batch_size),
        pdp.LambdaTransformer(extract_patches, buffer_size=batch_size),
        pdp.Chunker(chunk_size=batch_size, buffer_size=3),
        pdp.LambdaTransformer(pdp.combine_batches, buffer_size=buffer_size)
    )
