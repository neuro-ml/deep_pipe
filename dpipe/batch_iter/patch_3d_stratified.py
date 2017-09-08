from functools import lru_cache, partial
from random import choice

import numpy as np

from dpipe import medim
import dpipe.externals.pdp.pdp as pdp


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


def make_3d_patch_stratified_iter(
        ids, load_x, load_y, *, batch_size, x_patch_sizes,
        y_patch_size, nonzero_fraction, buffer_size=10):
    x_patch_sizes = np.array(x_patch_sizes)
    y_patch_size = np.array(y_patch_size)
    max_patch_size = np.max([*x_patch_sizes, y_patch_size], axis=0)

    assert np.all(x_patch_sizes % 2 == 1) and np.all(y_patch_size % 2 == 1)

    spatial_dims = [-3, -2, -1]

    random_seq = iter(partial(choice, ids), None)

    @lru_cache(maxsize=len(ids))
    def load_patient(name):
        return Patient(name, load_x(name), load_y(name))

    @lru_cache(maxsize=len(ids))
    def find_cancer(patient: Patient):
        x = pad_channel_min(patient.x, max_patch_size // 2)
        print(x.shape)

        y = patient.y
        if len(patient.y.shape) == 3:
            y = np.pad(y, (max_patch_size // 2)[None, :].repeat()(2, axis=0),
                       mode='constant')
            mask = y > 0
        elif len(patient.y.shape) == 4:
            y = pad_channel_min(y, max_patch_size // 2)
            mask = np.any(y, axis=0)
        else:
            raise ValueError('wrong number of dimensions ')
        print('going to')
        conditional_centre_indices = medim.patch.get_conditional_center_indices(
            mask, patch_size=max_patch_size, spatial_dims=spatial_dims)
        return x, y, conditional_centre_indices

    @pdp.pack_args
    def extract_patch(x, y, conditional_center_indices):
        assert all([x.shape[i] == y.shape[i] for i in spatial_dims])
        if np.random.uniform() < nonzero_fraction:
            center_idx = choice(conditional_center_indices)
        else:
            center_idx = medim.patch.get_uniform_center_index(
                x_shape=np.array(x.shape), patch_size=max_patch_size,
                spatial_dims=spatial_dims)

        xs = [medim.patch.extract_patch(
                  x, center_idx=center_idx, spatial_dims=spatial_dims,
                  patch_size=patch_size
              )
              for patch_size in x_patch_sizes]

        y = medim.patch.extract_patch(
            y, center_idx=center_idx, patch_size=y_patch_size,
            spatial_dims=spatial_dims
        )

        return (*xs, y)

    return pdp.Pipeline(
        pdp.Source(random_seq, buffer_size=3),
        pdp.LambdaTransformer(load_patient, buffer_size=100),
        pdp.LambdaTransformer(find_cancer, buffer_size=100),
        pdp.LambdaTransformer(extract_patch, n_workers=4,
                              buffer_size=batch_size),
        pdp.Chunker(chunk_size=batch_size, buffer_size=3),
        pdp.LambdaTransformer(pdp.combine_batches, buffer_size=buffer_size)
    )
