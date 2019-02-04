"""Input/Output operations."""
import argparse
import json
import re
import os
from pathlib import Path
from typing import Union

import numpy
import numpy as np

PathLike = Union[Path, str]


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


def load_experiment_test_pred(identifier, experiment_path):
    ep = Path(experiment_path)
    for f in os.listdir(ep):
        if os.path.isdir(ep / f):
            try:
                return load_pred(identifier, ep / f / 'test_predictions')
            except FileNotFoundError as e:
                print(e)
                pass
    else:
        raise FileNotFoundError('No prediction found')


def load_image(path: PathLike):
    """
    Load an image located at ``path``.
    The following extensions are supported:
        npy, tif, hdr, img, nii, nii.gz
    """
    if path.endswith('.npy'):
        return np.load(path)
    if path.endswith(('.nii', '.nii.gz', '.hdr', '.img')):
        import nibabel as nib
        return nib.load(path).get_data()
    if path.endswith('.tif'):
        from PIL import Image
        with Image.open(path) as image:
            return np.asarray(image)
    if path.endswith(('.png', '.jpg')):
        from imageio import imread
        return imread(path)
    raise ValueError(f"Couldn't read image from path: {path}. Unknown file extension.")


def load_json(path: PathLike):
    """Load the contents of a json file."""
    with open(path, 'r') as f:
        return json.load(f)


class NumpyEncoder(json.JSONEncoder):
    """A json encoder for numpy arrays and scalars."""

    def default(self, o):
        if isinstance(o, (np.generic, np.ndarray)):
            return o.tolist()
        return super().default(o)


def dump_json(value, path: PathLike, *, indent: int = None):
    """Dump a json-serializable object to a json file."""
    # TODO: probably should add makedirs here
    with open(path, 'w') as f:
        json.dump(value, f, indent=indent, cls=NumpyEncoder)
        return value


CONSOLE_ARGUMENT = re.compile(r'^--[^\d\W]\w*$')


class ConsoleArguments:
    """A class that simplifies access to console arguments."""

    def __init__(self):
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()[1]
        # allow for positional arguments:
        while args and not CONSOLE_ARGUMENT.match(args[0]):
            args = args[1:]

        self._args = {}
        for arg, value in zip(args[::2], args[1::2]):
            if not CONSOLE_ARGUMENT.match(arg):
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
