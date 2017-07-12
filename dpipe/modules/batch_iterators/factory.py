from itertools import islice
from contextlib import suppress
from abc import abstractmethod, ABC


class BatchIter(ABC):
    @abstractmethod
    def __iter__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class BatchIterFin(BatchIter):
    def __init__(self, get_batch_iter):
        self.get_batch_iter = get_batch_iter

    def __iter__(self):
        return iter(self.get_batch_iter())


class BatchIterInf(BatchIter):
    def __init__(self, get_batch_iter, n_iters_per_batch):
        self.batch_iter = get_batch_iter()
        self.n_iters_per_batch = n_iters_per_batch

    def __iter__(self):
        return islice(self.batch_iter, self.n_iters_per_batch)

    def __enter__(self):
        with suppress(AttributeError):
            self.batch_iter.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        with suppress(AttributeError):
            return self.batch_iter.__exit__(exc_type, exc_val, exc_tb)


def build_batch_iter(get_batch_iter, n_iters_per_batch) -> BatchIter:
    if n_iters_per_batch is None:
        return BatchIterFin(get_batch_iter)
    else:
        return BatchIterInf(get_batch_iter, n_iters_per_batch)
