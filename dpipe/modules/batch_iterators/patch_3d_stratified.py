from functools import lru_cache
from random import choice

import numpy as np

from ..datasets import Dataset
from .utils import combine_batch
from dpipe import medim
import dpipe.external.pdp.pdp as pdp


class Patient:
    def __init__(self, patient_id, mscan, msegm):
        self.patient_id = patient_id
        self.mscan = mscan
        self.msegm = msegm

    def __hash__(self):
        return hash(self.patient_id)


def make_3d_patch_stratified_iter(
        ids, dataset: Dataset, *, batch_size, x_patch_sizes, y_patch_size,
        nonzero_fraction, buffer_size=10):
    x_patch_sizes = [np.array(x_patch_size) for x_patch_size in x_patch_sizes]
    y_patch_size = np.array(y_patch_size)
    spatial_dims = [-3, -2, -1]

    def make_random_seq(l):
        while True:
            yield choice(l)

    def load_data(patient_name):
        mscan = dataset.load_x(patient_name)
        msegm = dataset.load_y(patient_name)

        return Patient(patient_name, mscan, msegm)

    @lru_cache(maxsize=len(ids))
    def find_cancer(patient: Patient):
        conditional_centre_indices = medim.patch.get_conditional_center_indices(
            np.any(patient.msegm, axis=0), patch_size=y_patch_size,
            spatial_dims=spatial_dims)

        return patient.mscan, patient.msegm, conditional_centre_indices

    @pdp.pack_args
    def extract_patch(mscan, msegm, conditional_center_indices):
        cancer_type = np.random.choice(
            [True, False], p=[nonzero_fraction, 1 - nonzero_fraction])
        if cancer_type:
            center_idx = choice(conditional_center_indices)
        else:
            center_idx = medim.patch.get_uniform_center_index(
                x_shape=np.array(mscan.shape), patch_size=y_patch_size,
                spatial_dims=spatial_dims)

        xs = [medim.patch.extract_patch(
            mscan, center_idx=center_idx, spatial_dims=spatial_dims,
            patch_size=patch_size)
            for patch_size in x_patch_sizes]

        y = medim.patch.extract_patch(
            msegm, center_idx=center_idx, patch_size=y_patch_size,
            spatial_dims=spatial_dims)

        return (*xs, y)

    return pdp.Pipeline(
        pdp.Source(make_random_seq(ids), buffer_size=3),
        pdp.LambdaTransformer(load_data, buffer_size=3),
        pdp.LambdaTransformer(find_cancer, buffer_size=3),
        pdp.LambdaTransformer(extract_patch, buffer_size=batch_size),
        pdp.Chunker(chunk_size=batch_size, buffer_size=3),
        pdp.LambdaTransformer(combine_batch, buffer_size=buffer_size)
    )
