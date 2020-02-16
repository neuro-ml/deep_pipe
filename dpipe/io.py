"""
Input/Output operations.

All the loading functions have the interface ``load(path, **kwargs)``
where ``kwargs`` are loader-specific keyword arguments.

Similarly, all the saving functions have the interface ``save(value, path, **kwargs)``.
"""
import argparse
import json
import pickle
import re
import os
from pathlib import Path
from typing import Union, Callable

import numpy as np
import pandas as pd

__all__ = [
    'PathLike', 'ConsoleArguments', 'load_or_create',
    'load', 'save',
    'load_json', 'save_json',
    'load_pickle', 'save_pickle',
    'load_numpy', 'save_numpy',
]

PathLike = Union[Path, str]


def load_pred(identifier, predictions_path):
    """
    Loads the prediction numpy tensor with specified id.

    Parameters
    ----------
    identifier: str, int
        id to load, could be either the file name ends with ``.npy``
    predictions_path: str
        path where to load prediction from

    Returns
    -------
    prediction: numpy.float32
    """
    if isinstance(identifier, int):
        _id = str(identifier) + '.npy'
    elif isinstance(identifier, str):
        if identifier.endswith('.npy'):
            _id = identifier
        else:
            _id = identifier + '.npy'
    else:
        raise TypeError(f'`identifier` should be either `int` or `str`, {type(identifier)} given')

    return np.float32(np.load(os.path.join(predictions_path, _id)))


def load_experiment_test_pred(identifier, experiment_path):
    ep = Path(experiment_path)
    for f in os.listdir(ep):
        if os.path.isdir(ep / f):
            try:
                return load_pred(identifier, ep / f / 'test_predictions')
            except FileNotFoundError:
                pass
    else:
        raise FileNotFoundError('No prediction found')


def load(path: PathLike, **kwargs):
    """
    Load a file located at ``path``.
    ``kwargs`` are format-specific keyword arguments.

    The following extensions are supported:
        npy, tif, hdr, img, nii, nii.gz, json, mhd, csv, txt, pickle, pkl
    """
    name = Path(path).name

    if name.endswith('.npy'):
        return load_numpy(path, **kwargs)
    if name.endswith(('.nii', '.nii.gz', '.hdr', '.img')):
        import nibabel as nib
        return nib.load(path, **kwargs).get_data()
    if name.endswith(('.png', '.jpg', '.tif')):
        from imageio import imread
        return imread(path, **kwargs)
    if name.endswith('.json'):
        return load_json(path, **kwargs)
    if name.endswith('.csv'):
        return pd.read_csv(path, **kwargs)
    if name.endswith(('.pkl', '.pickle')):
        return load_pickle(path, **kwargs)
    if name.endswith('.txt'):
        with open(path, mode='r', **kwargs) as file:
            return file.read()
    if name.endswith('.mhd'):
        from SimpleITK import ReadImage
        return ReadImage(name, **kwargs)

    raise ValueError(f'Couldn\'t read file "{path}". Unknown extension.')


def save(value, path: PathLike, **kwargs):
    """
    Save ``value`` to a file located at ``path``.
    ``kwargs`` are format-specific keyword arguments.

    The following extensions are supported:
        npy, tif, hdr, img, nii, nii.gz, json, txt, pickle, pkl
    """
    name = Path(path).name

    if name.endswith('.npy'):
        save_numpy(value, path, **kwargs)
    elif name.endswith(('.nii', '.nii.gz', '.hdr', '.img')):
        import nibabel as nib
        nib.save(value, path, **kwargs)
    elif name.endswith(('.png', '.jpg', '.tif')):
        from imageio import imsave
        imsave(path, value, **kwargs)
    elif name.endswith('.json'):
        save_json(value, path, **kwargs)
    elif name.endswith(('.pkl', '.pickle')):
        save_pickle(value, path, **kwargs)
    elif name.endswith('.txt'):
        with open(path, mode='w', **kwargs) as file:
            file.write(value)

    else:
        raise ValueError(f'Couldn\'t write to file "{path}". Unknown extension.')


def load_json(path: PathLike):
    """Load the contents of a json file."""
    with open(path, 'r') as f:
        return json.load(f)


class NumpyEncoder(json.JSONEncoder):
    """A json encoder with support for numpy arrays and scalars."""

    def default(self, o):
        if isinstance(o, (np.generic, np.ndarray)):
            return o.tolist()
        return super().default(o)


def save_json(value, path: PathLike, *, indent: int = None):
    """Dump a json-serializable object to a json file."""
    with open(path, 'w') as f:
        json.dump(value, f, indent=indent, cls=NumpyEncoder)


def save_numpy(value, path: PathLike, *, allow_pickle: bool = True, fix_imports: bool = True):
    """A wrapper around ``np.save`` that matches the interface ``save(what, where)``."""
    np.save(path, value, allow_pickle=allow_pickle, fix_imports=fix_imports)


def load_numpy(path: PathLike, *, allow_pickle: bool = True, fix_imports: bool = True):
    """A wrapper around ``np.load`` with ``allow_pickle`` set to True by default."""
    return np.load(path, allow_pickle=allow_pickle, fix_imports=fix_imports)


def save_pickle(value, path: PathLike):
    """Pickle a ``value`` to ``path``."""
    with open(path, 'wb') as file:
        pickle.dump(value, file)


def load_pickle(path: PathLike):
    """Load a pickled value from ``path``."""
    with open(path, 'rb') as file:
        return pickle.load(file)


def load_or_create(path: PathLike, create: Callable, *args,
                   save: Callable = save, load: Callable = load, **kwargs):
    """
    ``load`` a file from ``path`` if it exists.
    Otherwise ``create`` the value, ``save`` it to ``path``, and return it.

    ``args`` and ``kwargs`` are passed to ``create`` as additional arguments.
    """
    try:
        return load(path)
    except FileNotFoundError:
        pass

    value = create(*args, **kwargs)
    save(value, path)
    return value


class ConsoleArguments:
    """A class that simplifies access to console arguments."""

    _argument_pattern = re.compile(r'^--[^\d\W]\w*$')

    def __init__(self):
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()[1]
        # allow for positional arguments:
        while args and not self._argument_pattern.match(args[0]):
            args = args[1:]

        self._args = {}
        for arg, value in zip(args[::2], args[1::2]):
            if not self._argument_pattern.match(arg):
                raise ValueError(f'Invalid console argument: {arg}')
            self._args[arg[2:]] = value

    def __getattr__(self, name: str):
        """Get the console argument with the corresponding ``name``."""
        try:
            return self._args[name]
        except KeyError:
            raise AttributeError(f'Console argument {name} not provided.') from None

    def __call__(self, **kwargs):
        """
        Get a corresponding console argument, or return the default value if not provided.

        Parameters
        ----------
        kwargs:
            contains a single (key: value) pair, where `key` is the argument's name and `value` is its default value.

        Examples
        --------
        >>> console = ConsoleArguments()
        >>> # return `data_path` or '/some/default/path', if not provided
        >>> x = console(data_path='/some/default/path')
        """
        if len(kwargs) != 1:
            raise ValueError(f'This method takes exactly one argument, but {len(kwargs)} were passed.')
        name, value = list(kwargs.items())[0]
        return self._args.get(name, value)
