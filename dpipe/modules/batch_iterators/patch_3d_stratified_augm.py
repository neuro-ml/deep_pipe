from functools import lru_cache, partial
from random import choice

from itertools import product

import numpy as np
import scipy
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import zoom, map_coordinates
from scipy.ndimage.filters import gaussian_filter


from ..datasets import Dataset
from .utils import combine_batch
from dpipe import medim
import dpipe.external.pdp.pdp as pdp


class Patient:
    def __init__(self, patient_id, x, y):
        self.patient_id = patient_id
        self.x = x
        self.y = y

    def __hash__(self):
        return hash(self.patient_id)


def make_3d_augm_patch_stratified_iter(
        ids, dataset: Dataset, *, batch_size, x_patch_sizes, y_patch_size,
        nonzero_fraction, buffer_size=10):
    x_patch_sizes = [np.array(x_patch_size) for x_patch_size in x_patch_sizes]
    y_patch_size = np.array(y_patch_size)
    spatial_dims = [-3, -2, -1]

    random_seq = iter(partial(choice, ids), None)

    def load_patient(name):
        return Patient(name, dataset.load_x(name), dataset.load_y(name))

    @lru_cache(maxsize=len(ids))
    def find_cancer(patient: Patient):
        if patient.y.ndim == 4:
            mask = np.any(patient.y, axis=0)
        elif patient.y.ndim == 3:
            mask = patient.y > 0
        else:
            raise ValueError('y number of dimensions '
                             'was {}, expected 3 or 4'.format(patient.y.ndim))
        conditional_centre_indices = medim.patch.get_conditional_center_indices(
            mask, patch_size=y_patch_size, spatial_dims=spatial_dims)

        return patient.x, patient.y, conditional_centre_indices

    @pdp.pack_args
    def extract_big_patch(x, y, conditional_center_indices):
        cancer_type = np.random.choice(
            [True, False], p=[nonzero_fraction, 1 - nonzero_fraction])
        if cancer_type:
            center_idx = choice(conditional_center_indices)
        else:
            center_idx = medim.patch.get_uniform_center_index(
                x_shape=np.array(x.shape), patch_size=y_patch_size,
                spatial_dims=spatial_dims)

        xs = medim.patch.extract_patch(
            x, center_idx=center_idx, spatial_dims=spatial_dims,
            patch_size=x_patch_sizes[0])

        y = medim.patch.extract_patch(
            y, center_idx=center_idx, patch_size=y_patch_size,
            spatial_dims=spatial_dims)

        return (xs, y)

    def _scale_crop(x):
        shape = np.array(x.shape)
        x = zoom(x, scale, order=0)
        new_shape = np.array(x.shape)
        delta = shape - new_shape

        d_pad = np.maximum(0, delta)
        d_pad = list(zip(d_pad // 2, (d_pad + 1) // 2))
        d_slice = np.maximum(0, -delta)
        d_slice = zip(d_slice // 2, new_shape - (d_slice + 1) // 2)
        d_slice = [slice(x, y) for x, y in d_slice]

        x = x[d_slice]
        x = np.pad(x, d_pad, mode='constant')
        return x

    def _rotate(x, order=3, theta=0, alpha=0):
        x = rotate(x, theta, axes=(len(x.shape) - 2, len(x.shape) - 3),
                   reshape=False, order=order)
        x = rotate(x, alpha, axes=(len(x.shape) - 2, len(x.shape) - 1),
                   reshape=False, order=order)
        return x

    def augment(x: np.ndarray, y: np.ndarray):
        """
        Data random augmentation include scaling, rotation,
        axes flipping.
        __________________
        Say thanks to Max!
        """
        scipy.random.seed()

        # scale = np.random.normal(1, 0.1, size=3)
        alpha, theta = np.random.normal(0, 9, size=2)
        alpha = 0

        for i in range(1, len(x.shape) - 1):
            if np.random.binomial(1, .5):
                x = np.flip(x, -i)
                y = np.flip(y, -i)

        # x = np.array([_scale_crop(i) for i in x])
        # y = _scale_crop(y[0])[np.newaxis]

        x = _rotate(x, 3, theta, alpha)
        y = _rotate(y, 0, theta, alpha)

        #     if np.random.binomial(1, .5):
        #         t = np.random.choice([-90, 0, 90])
        #         a = np.random.choice([-90, 0, 90])
        #         x = _rotate(x, 3, t, a)
        #         y = _rotate(y, 3, t, a)

        x = np.array([i * np.random.normal(1, 0.35) for i in x])
        return x, y


    @pdp.pack_args
    def augmentation(x_big, y):
        return augment(x_big, y)

    @pdp.pack_args
    def extract_patch(x_big, y):

        center_idx = np.array(x_big.shape)[spatial_dims] // 2 + np.array(
            x_big.shape)[spatial_dims] % 2

        xs = [x_big] + [medim.patch.extract_patch(
            x_big, center_idx=center_idx, spatial_dims=spatial_dims,
            patch_size=patch_size)
            for patch_size in x_patch_sizes[1:]]

        # y = medim.patch.extract_patch(
        #     y, center_idx=center_idx, patch_size=y_patch_size,
        #     spatial_dims=spatial_dims)

        return (*xs, y)


    return pdp.Pipeline(
        pdp.Source(random_seq, buffer_size=3),
        pdp.LambdaTransformer(load_patient, buffer_size=3),
        pdp.LambdaTransformer(find_cancer, n_workers=8, buffer_size=3),
        pdp.LambdaTransformer(extract_big_patch,
                              n_workers=8, buffer_size=batch_size),
        pdp.LambdaTransformer(augmentation, n_workers=8,
                              buffer_size=0),
        pdp.LambdaTransformer(extract_patch, n_workers=8,
                              buffer_size=batch_size),
        pdp.Chunker(chunk_size=batch_size, buffer_size=3),
        pdp.LambdaTransformer(combine_batch, buffer_size=buffer_size)
    )
