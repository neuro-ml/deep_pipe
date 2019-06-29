"""Contains a few more sophisticated commands that are usually accessed directly inside configs."""

import os
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np
from tqdm import tqdm

from dpipe.medim.io import save_json
from dpipe.medim.itertools import collect


def np_filename2id(filename):
    *rest, extension = filename.split('.')
    assert extension == 'npy', f'Expected npy file, got {extension} from {filename}'
    return '.'.join(rest)


def transform(input_path, output_path, transform_fn):
    os.makedirs(output_path)

    for f in tqdm(os.listdir(input_path)):
        np.save(os.path.join(output_path, f), transform_fn(np.load(os.path.join(input_path, f))))


@collect
def load_from_folder(path: str):
    """Yields (id, object) pairs loaded from ``path``."""
    # TODO: generalize with a loader
    for filename in sorted(os.listdir(path)):
        yield np_filename2id(filename), np.load(os.path.join(path, filename))


def map_ids_to_disk(func: Callable[[str], object], ids: Iterable[str], output_path: str,
                    exist_ok: bool = False, save: Callable = np.save):
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

        save(output, value)
        # saving some memory
        del value


def predict(ids, output_path, load_x, predict_fn, exist_ok=False, save=np.save):
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
