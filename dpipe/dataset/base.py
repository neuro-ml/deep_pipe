from abc import ABCMeta
from functools import wraps
from typing import Tuple


class AbstractAttribute:
    def __init__(self, description: str):
        self.description = description

    def __repr__(self):
        return self.description


class ABCAttributesMeta(ABCMeta):
    def __new__(mcs, *args, **kwargs):
        cls = super().__new__(mcs, *args, **kwargs)
        initialize = cls.__init__

        @wraps(initialize)
        def __init__(self, *args_, **kwargs_):
            return_value = initialize(self, *args_, **kwargs_)

            # the check must be performed only after own __init__ is called
            if type(self) is cls:
                missing = []
                for name in dir(self):
                    value = getattr(self, name)
                    if isinstance(value, AbstractAttribute) or value is AbstractAttribute:
                        missing.append(name)
                if missing:
                    raise AttributeError(f'Class "{cls.__name__}" requires the following attributes '
                                         f'which are not defined during init: {", ".join(missing)}.')
            return return_value

        cls.__init__ = __init__
        return cls


class Dataset(metaclass=ABCAttributesMeta):
    """
    Interface for datasets.

    Its subclasses must define the `ids` attribute - a tuple of identifiers,
    one for each dataset entry, as well as methods for loading an entry by its identifier.

    Attributes
    ----------
    ids: a tuple of identifiers, one for each dataset entry.
    """
    ids: Tuple[str] = AbstractAttribute
