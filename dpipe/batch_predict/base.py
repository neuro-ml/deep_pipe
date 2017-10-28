from abc import ABC, abstractmethod


class BatchPredict(ABC):
    @abstractmethod
    def validate(self, *inputs, validate_fn):
        pass

    @abstractmethod
    def predict(self, *inputs, predict_fn):
        pass
