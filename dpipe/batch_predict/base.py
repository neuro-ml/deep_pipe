from abc import ABC, abstractmethod


class BatchPredict(ABC):
    @abstractmethod
    def validate(self, x, y, *, validate_fn):
        pass

    @abstractmethod
    def predict(self, x, *, predict_fn):
        pass
