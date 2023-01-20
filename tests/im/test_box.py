import unittest

from dpipe.im.box import *


class TestBox(unittest.TestCase):
    def test_make_box(self):
        box = make_box_(((0, 0), (12, 123)))
        with self.assertRaisesRegex(ValueError, 'destination is read-only'):
            box[0, 0] = 1

    def test_broadcast_box(self):
        spatial_box = make_box_(((2, 3), (5, 5)))
        shape = np.array([3, 10, 10, 10])
        spatial_dims = (-3, -2)

        np.testing.assert_equal(broadcast_box(spatial_box, shape, spatial_dims),
                                ((0, 2, 3, 0), (3, 5, 5, 10)))

    def test_get_containing_box(self):
        shape = (3, 4, 10, 12)
        np.testing.assert_equal(get_containing_box(shape), [[0, 0, 0, 0], shape])

    def test_return_box(self):
        box = returns_box(lambda: ((0, 0), (12, 123)))()
        with self.assertRaisesRegex(ValueError, 'destination is read-only'):
            box[0, 0] = 1

    def test_limit_full_box(self):
        limit = (1, 3, 4, 5)
        box = (tuple([0] * len(limit)), tuple(limit))
        np.testing.assert_array_equal(limit_box(box, limit), box)

    def test_limit_box(self):
        limit = (10, 10, 10)
        box = ((0, -1, -10), (10, 100, 17))
        np.testing.assert_array_equal(limit_box(box, limit), ((0, 0, 0), (10, 10, 10)))

    def test_get_box_padding(self):
        limit = np.array((10, 10, 10))
        box = np.array(((0, -1, -10), (10, 100, 17)))
        np.testing.assert_array_equal(get_box_padding(box, limit).T, ((0, 1, 10), (0, 90, 7)))

    def test_add_margin(self):
        box = np.array(((0, -1), (10, 100)))
        margin = 10
        np.testing.assert_array_equal(add_margin(box, margin), ((-10, -11), (20, 110)))
        margin = [1, 10]
        np.testing.assert_array_equal(add_margin(box, margin), ((-1, -11), (11, 110)))

    def test_get_centered_box(self):
        box_size = np.array((2, 3))
        center = np.array([5, 6])
        np.testing.assert_array_equal(get_centered_box(center=center, box_size=box_size),
                                      ((4, 5), (6, 8)))

        limit = np.array((15, 16))
        center = limit // 2
        start, stop = get_centered_box(center, limit)
        np.testing.assert_array_equal(start, 0)
        np.testing.assert_array_equal(stop, limit)

    def test_mask2bounding_box(self):
        mask = np.zeros((10, 10))
        mask[2, 3] = mask[4, 5] = True
        np.testing.assert_array_equal(mask2bounding_box(mask), ((2, 3), (5, 6)))
