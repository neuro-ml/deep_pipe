import unittest

import numpy as np
from .patch import extract_patch, sample_box_center_uniformly, get_random_patch
from .box import make_box_, get_centered_box


class TestPatch(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1, 2, 2, 3],
                           [2, 3, 3, 2],
                           [2, 3, 5, 2],
                           [4, 2, 2, 5]])

    def test_extract_patch_no_padding(self):
        x = extract_patch(self.x, box=make_box_(((0, 1), (4, 4))))
        np.testing.assert_array_equal(x, np.array([
            [2, 2, 3],
            [3, 3, 2],
            [3, 5, 2],
            [2, 2, 5]
        ]))

    def test_extract_padding(self):
        x = extract_patch(self.x, box=make_box_(((0, 2), (4, 5))), padding_values=[[1], [2], [3], [4]])
        np.testing.assert_array_equal(x, np.array([
            [2, 3, 1],
            [3, 2, 2],
            [5, 2, 3],
            [2, 5, 4]
        ]))

    def test_get_uniform_center_index(self):
        shape = self.x.shape
        box_size = np.array([4, 3])

        hits = {(0, 0): 0, (0, 1): 0}

        n = 1000
        for _ in range(n):
            center = sample_box_center_uniformly(shape, box_size=box_size)
            box = get_centered_box(center, box_size=box_size)
            np.testing.assert_equal(box[1] - box[0], box_size)
            hits[tuple(box[0])] += 1
        self.assertIn(hits[(0, 0)], range(400, 600))


class TestRandomPatch(unittest.TestCase):
    def test_no_spatial_dims(self):
        shape = (3, 4, 10)
        x = np.empty(shape)
        patch = get_random_patch(x, [2, 2])
        self.assertEqual(patch.shape, (3, 2, 2))

    def test_spatial_dims(self):
        shape = (3, 4, 10)
        x = np.empty(shape)
        patch = get_random_patch(x, [2, 2], spatial_dims=[0, 2])
        self.assertEqual(patch.shape, (2, 4, 2))
