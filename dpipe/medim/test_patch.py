import unittest

import numpy as np
from .patch import find_patch_size, find_patch_start_end_padding, \
    extract_patch, find_masked_patch_center_indices, sample_uniform_center_index


class TestPatch(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1, 2, 2, 3],
                           [2, 3, 3, 2],
                           [2, 3, 5, 2],
                           [4, 2, 2, 5]])
        self.patch_size = np.array([3])
        self.spatial_dims = [1]

    def test_find_patch_sizes(self):
        shape = np.array([3, 12, 13, 14])
        spatial_patch_size = np.array([3, 5, 7])
        patch_size = find_patch_size(shape, spatial_patch_size, [-3, -2, -1])
        np.testing.assert_array_equal(patch_size, [3, 3, 5, 7])

    def test_find_patch_start_end_padding(self):
        shape = np.array([3, 12, 13, 14])
        spatial_patch_size = np.array([3, 5, 7])
        spatial_centre = np.array([1, 1, 1])
        start, end, padding = find_patch_start_end_padding(
            shape, spatial_center_idx=spatial_centre,
            spatial_patch_size=spatial_patch_size, spatial_dims=[-3, -2, -1]
        )
        np.testing.assert_equal(start, [0, 0, 0, 0])
        np.testing.assert_equal(end, [3, 3, 4, 5])
        np.testing.assert_equal(padding, [[0, 0], [0, 0], [1, 0], [2, 0]])

    def test_extract_patch_no_padding(self):
        x = extract_patch(self.x, spatial_center_idx=[2],
                          spatial_patch_size=self.patch_size,
                          spatial_dims=self.spatial_dims)

        self.assertSequenceEqual(x.shape, (4, 3))
        np.testing.assert_array_equal(x, np.array([
            [2, 2, 3],
            [3, 3, 2],
            [3, 5, 2],
            [2, 2, 5]]
        ))

    def test_extract_padding(self):
        x = extract_patch(self.x, spatial_center_idx=np.array([3]),
                          spatial_patch_size=self.patch_size,
                          spatial_dims=self.spatial_dims,
                          padding_values=np.array([[1], [2], [3], [4]]))

        self.assertSequenceEqual(x.shape, (4, 3))
        np.testing.assert_array_equal(x, np.array([
            [2, 3, 1],
            [3, 2, 2],
            [5, 2, 3],
            [2, 5, 4]
        ]))

    def test_get_uniform_center_index(self):
        for _ in range(1000):
            c = sample_uniform_center_index(
                np.array(self.x.shape), spatial_patch_size=self.patch_size,
                spatial_dims=self.spatial_dims)
            self.assertTrue((c == [1]) or (c == [2]))

    def test_get_conditional_center_indices(self):
        c = find_masked_patch_center_indices(
            np.any(self.x > 4, axis=0), patch_size=self.patch_size
        )
        np.testing.assert_array_equal(c, [[2]])

