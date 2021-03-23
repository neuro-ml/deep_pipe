import tempfile
import unittest

import numpy as np

from dpipe.train.checkpoint import Checkpoints
from dpipe.train.policy import Exponential, DecreasingOnPlateau, Schedule


class TestCheckpoints(unittest.TestCase):
    @staticmethod
    def advance_policies(policies, current, epochs):
        for epoch in range(current, current + epochs):
            for policy in policies:
                policy.epoch_started(epoch)
                policy.epoch_finished(epoch, train_losses=np.random.uniform(0, .01, 100))

        return current + epochs

    @staticmethod
    def get_params(policies):
        return [policy.value for policy in policies]

    def test_policies(self):
        with tempfile.TemporaryDirectory() as tempdir:
            policies = {
                'some_weight': Exponential(1, .1, 10, False),
                'lr': DecreasingOnPlateau(initial=1, multiplier=.1, patience=4, rtol=.02, atol=.02),
                'scheduled': Schedule.constant_multiplier(1, .1, [3, 5, 10])
            }
            manager = Checkpoints(tempdir, policies)

            current = 0
            for epoch in range(10):
                current = self.advance_policies(policies.values(), current, 4)
                manager.save(epoch)

                params = self.get_params(policies.values())
                current = self.advance_policies(policies.values(), current, 4)
                assert any(old != new for old, new in zip(params, self.get_params(policies.values())))

                manager.restore()
                assert params == self.get_params(policies.values())

    def test_sequence(self):
        with tempfile.TemporaryDirectory() as tempdir:
            policies = [
                Exponential(1, .1, 10, False),
                Exponential(1, .1, 5, True),
                DecreasingOnPlateau(initial=1, multiplier=.1, patience=4, rtol=.02, atol=.02),
                Schedule.constant_multiplier(1, .1, [3, 5, 10]),
            ]
            manager = Checkpoints(tempdir, policies)

            current = 0
            for epoch in range(10):
                current = self.advance_policies(policies, current, 4)
                manager.save(epoch)

                params = self.get_params(policies)
                current = self.advance_policies(policies, current, 4)
                assert any(old != new for old, new in zip(params, self.get_params(policies)))

                manager.restore()
                assert params == self.get_params(policies)
