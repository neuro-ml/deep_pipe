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
    """
    Loads the contents of a json file.

    Parameters
    ----------
    path: str
        path to the file

    Returns
    -------
    json_type
    """
    with open(path, 'r') as f:
        return json.load(f)


console_argument = re.compile(r'^--[^\d\W]\w*$')


class ConsoleArguments:
    """
    A class that simplifies the access to console arguments.
    """

    def __init__(self):
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()[1]
        # allow for positional arguments:
        while args and not console_argument.match(args[0]):
            args = args[1:]

        self.args = {}
        for arg, value in zip(args[::2], args[1::2]):
            if not console_argument.match(arg):
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
            raise ValueError(f'This method takes exactly one argument, '
                             f'but {len(kwargs)} were passed.')
        name = list(kwargs.keys())[0]
        return self.args.get(name, kwargs[name])
