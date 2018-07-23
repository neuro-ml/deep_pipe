import unittest

import numpy as np

from .patch import sample_box_center_uniformly, extract_patch, get_random_patch, get_random_patch_start_stop
from .box import make_box_, get_centered_box
from .test_utils import get_random_tuple


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
        x = np.empty((3, 4, 10))
        patch = get_random_patch(x, [2, 2])
        self.assertEqual(patch.shape, (3, 2, 2))

    def test_spatial_dims(self):
        x = np.empty((3, 4, 10))
        patch = get_random_patch(x, [2, 2], spatial_dims=[0, 2])
        self.assertEqual(patch.shape, (2, 4, 2))

    def test_random_path_start_stop(self):
        for _ in range(1000):
            size = np.random.randint(1, 6)
            shape = get_random_tuple(5, 20, size)
            kernel_size = get_random_tuple(1, min(shape) + 1, size)

            with self.subTest(shape=shape, kernel_size=kernel_size):
                start, stop = get_random_patch_start_stop(shape, kernel_size)
                np.testing.assert_array_compare(np.less_equal, 0, start)
                np.testing.assert_array_compare(np.less_equal, start, stop)
                np.testing.assert_array_compare(np.less_equal, stop, shape)

        with self.assertRaises(ValueError):
            get_random_patch_start_stop([3], [4])
