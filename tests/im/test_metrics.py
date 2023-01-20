import unittest
import numpy as np
import torch

from dpipe.im.metrics import cross_entropy_with_logits


class TestMetrics(unittest.TestCase):
    def test_cross_entropy(self):
        for _ in range(5):
            x = np.random.randn(40, 5, 10, 120) * 10
            y = np.random.randint(0, 5, (40, 10, 120))

            np.testing.assert_almost_equal(
                cross_entropy_with_logits(y, x, reduce=None),
                torch.nn.functional.cross_entropy(
                    torch.from_numpy(x), torch.from_numpy(y), reduction='none'
                ).data.numpy()
            )

        self.assertEqual(cross_entropy_with_logits(y, np.moveaxis(x, 1, -1), axis=-1), cross_entropy_with_logits(y, x))
