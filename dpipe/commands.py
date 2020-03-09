"""Contains a few more sophisticated commands that are usually accessed directly inside configs."""
import os
import shutil
import atexit
from pathlib import Path
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np
from tqdm import tqdm

from .io import save_json, save_numpy, load, PathLike
from .medim.itertools import collect


def populate(path: PathLike, func: Callable, *args, **kwargs):
    """
    Call ``func`` with ``args`` and ``kwargs`` if ``path`` doesn't exist.

    Examples
    --------
    >>> populate('metrics.json', save_metrics, targets, predictions)
    # if `metrics.json` doesn't exist, the following call will be performed:
    >>> save_metrics(targets, predictions)

    Raises
    ------
    FileNotFoundError: if after calling ``func`` the ``path`` still doesn't exist.
    """

    def flush(message):
        print(f'\n>>> {message}', flush=True)

    path = Path(path)
    if path.exists():
        flush(f'Nothing to be done, "{path}" already exists.')
        return

    try:
        flush(f'Running command to generate "{path}".')
        func(*args, **kwargs)
    except BaseException as e:
        if path.exists():
            shutil.rmtree(path)
        raise RuntimeError('An exception occurred. The outputs were cleaned up.\n') from e

    if not path.exists():
        raise FileNotFoundError(f'The output was not generated: "{path}"')


def lock_dir(folder: PathLike = '.', lock: str = '.lock'):
    """
    Lock the given ``folder`` by generating a special lock file - ``lock``.

    Raises
    ------
    FileExistsError: if ``lock`` already exists, i.e. the folder is already locked.
    """
    lock = Path(folder) / lock
    if lock.exists():
        raise FileExistsError(f'Trying to lock directory {lock.resolve().parent}, but it is already locked.')

    lock.touch(exist_ok=False)
    atexit.register(os.remove, lock)


def np_filename2id(filename):
    *rest, extension = filename.split('.')
    assert extension == 'npy', f'Expected npy file, got {extension} from {filename}'
    return '.'.join(rest)


def transform(input_path, output_path, transform_fn):
    os.makedirs(output_path)

    for f in tqdm(os.listdir(input_path)):
        np.save(os.path.join(output_path, f), transform_fn(np.load(os.path.join(input_path, f))))


@collect
def load_from_folder(path: str, loader=load):
    """Yields (id, object) pairs loaded from ``path``."""
    for filename in sorted(os.listdir(path)):
        yield np_filename2id(filename), loader(os.path.join(path, filename))


def map_ids_to_disk(func: Callable[[str], object], ids: Iterable[str], output_path: str,
                    exist_ok: bool = False, save: Callable = save_numpy):
    """
    Apply ``func`` to each id from ``ids`` and save each output to ``output_path`` using ``save``.
    If ``exist_ok`` is True the existing files will be ignored, otherwise an exception is raised.
    """
    os.makedirs(output_path, exist_ok=exist_ok)

    for identifier in ids:
        output = os.path.join(output_path, f'{identifier}.npy')
        if exist_ok and os.path.exists(output):
            continue

        value = func(identifier)

        # To save disk space
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
            value = value.astype(np.float16)

        save(value, output)
        # saving some memory
        del value


def predict(ids, output_path, load_x, predict_fn, exist_ok=False, save=save_numpy):
    map_ids_to_disk(lambda identifier: predict_fn(load_x(identifier)), tqdm(ids), output_path, exist_ok, save)


def evaluate_aggregated_metrics(load_y_true, metrics: dict, predictions_path, results_path, exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    targets, predictions = [], []
    for identifier, prediction in tqdm(load_from_folder(predictions_path)):
        predictions.append(prediction)
        targets.append(load_y_true(identifier))

    for name, metric in metrics.items():
        save_json(metric(targets, predictions), os.path.join(results_path, name + '.json'), indent=0)


def evaluate_individual_metrics(load_y_true, metrics: dict, predictions_path, results_path, exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for identifier, prediction in tqdm(load_from_folder(predictions_path)):
        target = load_y_true(identifier)

        for metric_name, metric in metrics.items():
            results[metric_name][identifier] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)
