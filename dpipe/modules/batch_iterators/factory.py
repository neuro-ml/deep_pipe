from itertools import islice
from contextlib import suppress, contextmanager
from abc import abstractmethod, ABCMeta


@contextmanager
def build_contextmanager(o):
    yield o


def maybe_build_contextmanager(o):
    if hasattr(o, '__enter__'):
        return o
    else:
        return build_contextmanager(o)


class BatchIterFactory(metaclass=ABCMeta):
    @abstractmethod
    def __next__(self):
        pass

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class BatchIterFactoryFin(BatchIterFactory):
    def __init__(self, get_batch_iter):
        self.get_batch_iter = get_batch_iter

    def __next__(self):
        return maybe_build_contextmanager(self.get_batch_iter())


class BatchIterFactoryInf(BatchIterFactory):
    def __init__(self, get_batch_iter, n_iters_per_batch):
        self.inf_batch_iter = get_batch_iter()
        self.n_iters_per_batch = n_iters_per_batch

    def __next__(self):
        return build_contextmanager(
            islice(self.inf_batch_iter, self.n_iters_per_batch))

    def __enter__(self):
        with suppress(AttributeError):
            self.inf_batch_iter.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        with suppress(AttributeError):
            return self.inf_batch_iter.__exit__(exc_type, exc_val, exc_tb)


def build_batch_iter_factory(get_batch_iter, n_iters_per_batch) -> \
        BatchIterFactory:
    if n_iters_per_batch is None:
        return BatchIterFactoryFin(get_batch_iter)
    else:
        return BatchIterFactoryInf(get_batch_iter, n_iters_per_batch)
