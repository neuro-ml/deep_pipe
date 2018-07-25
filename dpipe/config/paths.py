""""File with utilities to get library paths."""
import os

from numpy import deprecate

REPOSITORY_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
paths = {key: os.path.join(REPOSITORY_PATH, path) for key, path in {
    "do": "scripts/do.py"
}.items()}


@deprecate
def get_paths():
    return paths
