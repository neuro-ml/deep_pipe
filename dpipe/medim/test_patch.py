import unittest

import numpy as np
from .patch import find_patch_start_end_padding, \
    get_conditional_center_indices, get_uniform_center_index, extract_patch


class TestPatch(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1, 2, 2, 3],
                           [2, 3, 3, 2],
                           [2, 3, 5, 2],
                           [4, 2, 2, 5]])
        self.patch_size = np.array([3])
        self.spatial_dims = [1]

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

    @unittest.skip('wrong test')
    def test_extract_patch_zero_padding(self):
        x = extract_patch(self.x, spatial_center_idx=[3],
                          spatial_patch_size=self.patch_size,
                          spatial_dims=self.spatial_dims)

        self.assertEqual(x.shape, (4, 3))
        x_true = np.array([[2, 3, 0],
                           [3, 2, 0],
                           [5, 2, 0],
                           [2, 5, 0]])
        self.assertSequenceEqual(list(x.flatten()), list(x_true.flatten()))

    @unittest.skip('wrong test')
    def test_extract_patch(self):
        x = extract_patch(self.x, spatial_center_idx=[2], spatial_patch_size=self.patch_size,
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
