import unittest

import numpy as np
from .patch import extract_patch_zero_padding, get_conditional_center_indices,\
    get_uniform_center_index, extract_patch


class TestPatch(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1, 2, 2, 3],
                           [2, 3, 3, 2],
                           [2, 3, 5, 2],
                           [4, 2, 2, 5]])
        self.patch_size = np.array([3])
        self.spatial_dims = [1]

    def test_extract_patch_zero_padding(self):
        x = extract_patch_zero_padding(self.x, center_idx=[3],
                                       patch_size=self.patch_size,
                                       spatial_dims=self.spatial_dims)

        self.assertEqual(x.shape, (4, 3))
        x_true = np.array([[2, 3, 0],
                           [3, 2, 0],
                           [5, 2, 0],
                           [2, 5, 0]])
        self.assertSequenceEqual(list(x.flatten()), list(x_true.flatten()))

    def test_extract_patch(self):
        x = extract_patch(self.x, center_idx=[2], patch_size=self.patch_size,
                          spatial_dims=self.spatial_dims)

        self.assertEqual(x.shape, (4, 3))
        x_true = np.array([[2, 2, 3],
                           [3, 3, 2],
                           [3, 5, 2],
                           [2, 2, 5]])
        self.assertSequenceEqual(list(x.flatten()), list(x_true.flatten()))


    def test_get_conditional_center_indices(self):
        c = get_conditional_center_indices(np.any(self.x > 4, axis=0),
            patch_size=self.patch_size,
            spatial_dims=self.spatial_dims)

        self.assertSequenceEqual(list(c.flatten()), [2])

    def test_get_uniform_center_index(self):
        for _ in range(1000):
            c = get_uniform_center_index(
                np.array(self.x.shape), patch_size=self.patch_size,
                spatial_dims=self.spatial_dims)
            self.assertTrue(np.all(c == [1]) or np.all(c == [2]))
