from abc import abstractmethod, ABC
from contextlib import contextmanager

__all__ = ['BatchIter']


@contextmanager
def build_contextmanager(o):
    yield o


def maybe_build_contextmanager(o):
    """If input has no context manager on it's own, turn it into object with empty context manager."""
    if hasattr(o, '__enter__'):
        return o
    else:
        return build_contextmanager(o)


class BatchIter(ABC):
    """Interface for training functions, that unifies interface for infinite and finite batch generators.

    Examples
    --------
    >>> # BatchIter should be created from one of the implementations
    >>> batch_iter : BatchIter = None
    >>> with batch_iter:
    >>>     for epoch in range(10):
    >>>         for x, y in batch_iter():
    >>>             pass

    """

    @abstractmethod
    def __call__(self):
        pass

    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
