from typing import Dict, Sequence, Any
from dpipe.train.policy import Policy, ValuePolicy


class PolicyHandler(Policy):
    """
    A Class used to store and update parameters during training procedure.
    """

    def __init__(self, parameters: Dict):
        self._scalars = self.get_scalars(parameters)
        self._policies = self.get_policies(parameters)

    @property
    def policies(self):
        return self.get_policy_values(self._policies)

    @staticmethod
    def get_policy_values(policies):
        return {k: v.value for k, v in policies.items() if isinstance(v, ValuePolicy)}

    @staticmethod
    def get_policies(params: dict):
        return {k: v for k, v in params.items() if isinstance(v, Policy)}

    @staticmethod
    def get_scalars(params: dict):
        return {k: v for k, v in params.items() if not isinstance(v, Policy)}

    @staticmethod
    def broadcast_event(policies, method, *args, **kw):
        for name, policy in policies.items():
            getattr(policy, method.__name__)(*args, **kw)

    @property
    def current_values(self):
        return {**self._scalars, **self.get_policy_values(self._policies)}

    def epoch_started(self, epoch: int):
        self.broadcast_event(self._policies, Policy.epoch_started, epoch)

    def epoch_finished(self, epoch: int, train_losses: Sequence, metrics: dict = None, policies: dict = None):
        self.broadcast_event(self._policies, Policy.epoch_finished, epoch, train_losses)

    def train_step_started(self, epoch: int, iteration: int):
        self.broadcast_event(self._policies, Policy.train_step_started, epoch, iteration)

    def train_step_finished(self, epoch: int, iteration: int, loss: Any):
        self.broadcast_event(self._policies, Policy.train_step_finished, epoch, iteration, loss)

    def validation_started(self, epoch: int, train_losses: Sequence):
        self.broadcast_event(self._policies, Policy.validation_started, epoch, train_losses)
