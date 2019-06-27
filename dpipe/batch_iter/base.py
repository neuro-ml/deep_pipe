from abc import abstractmethod, ABC


class BatchIter(ABC):
    """
    Interface for batch iterators.

    References
    ----------
    `Pipeline`

    Examples
    --------
    >>> batch_iter : BatchIter
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
