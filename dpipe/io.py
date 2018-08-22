"""
Basic input-output operations. Useful inside config-files.
"""

import argparse
import json
import re
import os

import numpy as np


def load_pred(identifier, predictions_path):
    """
    Loads the prediction numpy tensor with specified id.

    Parameters
    ----------
    identifier: int
        id to load
    predictions_path: str
        path where to load prediction from

    Returns
    -------
    prediction: numpy.float32
    """
    return np.float32(np.load(os.path.join(predictions_path, f'{identifier}.npy')))


def load_json(path: str):
    """Load the contents of a json file."""
    with open(path, 'r') as f:
        return json.load(f)


class NumpyEncoder(json.JSONEncoder):
    """A json encoder for numpy arrays and scalars."""

    def default(self, o):
        if isinstance(o, (np.generic, np.ndarray)):
            return o.tolist()
        return super().default(o)


def dump_json(value, path: str, *, indent: int = None):
    """Dump a json-serializable object to a json file."""
    with open(path, 'w') as f:
        return json.dump(value, f, indent=indent, cls=NumpyEncoder)


CONSOLE_ARGUMENT = re.compile(r'^--[^\d\W]\w*$')


class ConsoleArguments:
    """A class that simplifies the access to console arguments."""

    def __init__(self):
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()[1]
        # allow for positional arguments:
        while args and not CONSOLE_ARGUMENT.match(args[0]):
            args = args[1:]

        self.args = {}
        for arg, value in zip(args[::2], args[1::2]):
            if not CONSOLE_ARGUMENT.match(arg):
                raise ValueError(f'Invalid console argument: {arg}')
            arg = arg[2:]
            try:
                value = int(value)
            except ValueError:
                pass
            self.args[arg] = value

    def __getattr__(self, name: str):
        """
        Get the console argument with the corresponding name

        Parameters
        ----------
        name: str
            argument's name

        Returns
        -------
        argument_value

        Raises
        ------
        AttributeError
        """
        try:
            return self.args[name]
        except KeyError:
            raise AttributeError(f'Console argument {name} not provided') from None

    def __call__(self, **kwargs):
        """
        Get a corresponding console argument, or return `default` if not provided.

        Parameters
        ----------
        kwargs:
            contains a single (key: value) pair, where `key` is the argument's name
            and `value` is its default value

        Examples
        --------
        >>> console = ConsoleArguments()
        >>> # return `data_path` or '/some/default/path', if not provided
        >>> x = console(data_path='/some/default/path')
        """
        if len(kwargs) != 1:
            raise ValueError(f'This method takes exactly one argument, but {len(kwargs)} were passed.')
        name = list(kwargs.keys())[0]
        return self.args.get(name, kwargs[name])
