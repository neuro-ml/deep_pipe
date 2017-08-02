from functools import lru_cache, partial
from random import choice

import numpy as np

import dpipe.externals.pdp.pdp as pdp
from dpipe import medim
from dpipe.datasets import Dataset
from .utils import combine_batch


class Patient:
    def __init__(self, patient_id, mscan, segm):
        self.patient_id = patient_id
        self.mscan = mscan
        self.segm = segm

    def __hash__(self):
        return hash(self.patient_id)


def make_3d_patch_stratified_iter(
        ids, dataset: Dataset, *, batch_size, x_patch_sizes, y_patch_size,
        nonzero_fraction, buffer_size=10):
    x_patch_sizes = [np.array(x_patch_size) for x_patch_size in x_patch_sizes]
    y_patch_size = np.array(y_patch_size)
    spatial_dims = [-3, -2, -1]

    random_seq = iter(partial(choice, ids), None)

    def load_patient(name):
        return Patient(name, dataset.load_mscan(name), dataset.load_segm(name))

    @lru_cache(maxsize=len(ids))
    def find_cancer(patient: Patient):
        mask = patient.segm > 0
        conditional_centre_indices = medim.patch.get_conditional_center_indices(
            mask, patch_size=y_patch_size, spatial_dims=spatial_dims)

        return patient.mscan, patient.segm, conditional_centre_indices

    @pdp.pack_args
    def extract_patch(x, y, conditional_center_indices):
        if np.random.uniform() < nonzero_fraction:
            center_idx = choice(conditional_center_indices)
        else:
            center_idx = medim.patch.get_uniform_center_index(
                x_shape=np.array(x.shape), patch_size=y_patch_size,
                spatial_dims=spatial_dims)

        xs = [medim.patch.extract_patch(
            x, center_idx=center_idx, spatial_dims=spatial_dims,
            patch_size=patch_size)
            for patch_size in x_patch_sizes]

        y = medim.patch.extract_patch(
            y, center_idx=center_idx, patch_size=y_patch_size,
            spatial_dims=spatial_dims)

        return (*xs, y)

    return pdp.Pipeline(
        pdp.Source(random_seq, buffer_size=3),
        pdp.LambdaTransformer(load_patient, buffer_size=3),
        pdp.LambdaTransformer(find_cancer, buffer_size=3),
        pdp.LambdaTransformer(extract_patch, buffer_size=batch_size),
        pdp.Chunker(chunk_size=batch_size, buffer_size=3),
        pdp.LambdaTransformer(combine_batch, buffer_size=buffer_size)
    )
