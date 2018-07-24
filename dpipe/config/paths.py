""""File with utilities to get library paths."""
import os

REPOSITORY_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
paths = {key: os.path.join(REPOSITORY_PATH, path) for key, path in {
    "do": "scripts/do.py"
}.items()}


def get_paths():
    return paths
