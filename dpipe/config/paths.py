""""File with utilities to get library paths."""
import os
import json


def get_paths():
    current_path = os.path.realpath(os.path.dirname(__file__))
    repository_path = os.path.join(current_path, '../..')
    with open(os.path.join(repository_path, 'paths.json')) as f:
        paths = json.load(f)

    prev_dir = os.getcwd()
    try:
        os.chdir(repository_path)
        for k in paths:
            paths[k] = os.path.realpath(paths[k])
    finally:
        os.chdir(prev_dir)

    return paths
