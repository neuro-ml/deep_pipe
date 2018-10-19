import tempfile
import unittest

import numpy as np

from dpipe.train.checkpoint import CheckpointManager
from dpipe.train.policy import Exponential, DecreasingOnPlateau, Schedule


class TestCheckpoints(unittest.TestCase):
    @staticmethod
    def advance_policies(policies, epochs):
        for _ in range(epochs):
            for policy in policies:
                for _ in range(100):
                    policy.step_finished(1)

                policy.epoch_finished(train_losses=np.random.uniform(0, .01, 100))

    @staticmethod
    def get_params(policies):
        return [(policy.value, policy.epoch, policy.step, policy.total_steps) for policy in policies]

    def test_policies(self):
        with tempfile.TemporaryDirectory() as tempdir:
            policies = {
                'some_weight': Exponential(1, .1, 10, False),
                'lr': DecreasingOnPlateau(initial=1, multiplier=.1, patience=4, rtol=.02, atol=.02),
                'scheduled': Schedule.constant_multiplier(1, .1, [3, 5, 10])
            }
            manager = CheckpointManager(tempdir, pickled_objects=policies)

            for _ in range(10):
                self.advance_policies(policies.values(), 4)
                manager.save()

                params = self.get_params(policies.values())
                self.advance_policies(policies.values(), 4)
                self.assertTrue(any(old != new for old, new in zip(params, self.get_params(policies.values()))))

                manager.restore()
                self.assertListEqual(params, self.get_params(policies.values()))
