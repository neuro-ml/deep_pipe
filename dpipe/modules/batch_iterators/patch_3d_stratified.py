from functools import lru_cache
from random import choice

import numpy as np

from ..datasets import Dataset
from dpipe import medim
from bdp import Pipeline, LambdaTransformer, Source, Chunker, pack_args


class Patient:
    def __init__(self, patient_id, mscan, msegm):
        self.patient_id = patient_id
        self.mscan = mscan
        self.msegm = msegm

    def __hash__(self):
        return hash(self.patient_id)


def make_3d_patch_stratified_iter(
        ids, dataset: Dataset, *, batch_size, x_patch_sizes, y_patch_size,
        nonzero_fraction):
    x_patch_sizes = [np.array(x_patch_size) for x_patch_size in x_patch_sizes]
    y_patch_size = np.array(y_patch_size)
    spatial_size = np.array(dataset.spatial_size)
    spatial_dims = [1, 2, 3]

    n_chans_mscan = dataset.n_chans_mscan
    n_chans_msegm = dataset.n_chans_msegm

    x_shape = np.array([n_chans_msegm, *spatial_size])

    def make_random_seq(l):
        while True:
            yield choice(l)

    def load_data(patient_name):
        mscan = dataset.load_x(patient_name)
        msegm = dataset.load_y(patient_name)

        return Patient(patient_name, mscan, msegm)

    @lru_cache(maxsize=len(dataset.patient_ids))
    def find_cancer(patient: Patient):
        conditional_centre_indices = medim.patch.get_conditional_center_indices(
            np.any(patient.msegm, axis=0), patch_size=y_patch_size,
            spatial_dims=spatial_dims)

        return patient.mscan, patient.msegm, conditional_centre_indices

    @pack_args
    def extract_patch(mscan, msegm, conditional_center_indices):
        cancer_type = np.random.choice(
            [True, False], p=[nonzero_fraction, 1 - nonzero_fraction])
        if cancer_type:
            i = np.random.randint(len(conditional_center_indices))
            center_idx = conditional_center_indices[i]
        else:
            center_idx = medim.patch.get_uniform_center_index(
                x_shape=x_shape, patch_size=y_patch_size,
                spatial_dims=spatial_dims)

        xs = [medim.patch.extract_patch(
            mscan, center_idx=center_idx, spatial_dims=spatial_dims,
            patch_size=patch_size)
            for patch_size in x_patch_sizes]

        y = medim.patch.extract_patch(
            msegm, center_idx=center_idx, patch_size=y_patch_size,
            spatial_dims=spatial_dims)

        return (*xs, y)

    outputs = [np.zeros((batch_size, n_chans_mscan, *ps), dtype=np.float32)
               for ps in x_patch_sizes] + \
              [np.zeros((batch_size, n_chans_msegm, *y_patch_size),
                        dtype=np.float32)]

    def combine_batch(inputs):
        n_sources = len(inputs[0])
        for s in range(n_sources):
            for b in range(batch_size):
                outputs[s][b] = inputs[b][s]
        return outputs

    return Pipeline(
        Source(make_random_seq(ids), buffer_size=10),
        LambdaTransformer(load_data, n_workers=1, buffer_size=10),
        LambdaTransformer(find_cancer, n_workers=1, buffer_size=50),
        LambdaTransformer(extract_patch, n_workers=4,
                          buffer_size=3 * batch_size),
        Chunker(chunk_size=batch_size, buffer_size=2),
        LambdaTransformer(combine_batch, n_workers=1, buffer_size=3)
    )
