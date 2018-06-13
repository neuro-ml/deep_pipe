"""
Contains a few more sophisticated commands
that are usually accessed via the `do.py` script.
"""

import json
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from dpipe.medim.metrics import dice_score as dice, multichannel_dice_score
from dpipe.medim.utils import load_by_ids
from dpipe.train.validator import evaluate as evaluate_fn


def np_filename2id(filename):
    *rest, extension = filename.split('.')
    assert extension == 'npy', f'Expected npy file, got {extension} from {filename}'
    return '.'.join(rest)


def train_model(train, model, save_model_path, restore_model_path=None, modify_state_fn=None):
    if restore_model_path is not None:
        model.load(restore_model_path, modify_state_fn=modify_state_fn)

    train()
    model.save(save_model_path)


def transform(input_path, output_path, transform_fn):
    os.makedirs(output_path)

    for f in tqdm(os.listdir(input_path)):
        np.save(os.path.join(output_path, f), transform_fn(np.load(os.path.join(input_path, f))))


def predict(ids, output_path, load_x, predict_fn, exist_ok=False):
    os.makedirs(output_path, exist_ok=exist_ok)

    for identifier in tqdm(ids):
        output = os.path.join(output_path, f'{identifier}.npy')
        if exist_ok and os.path.exists(output):
            continue

        x = load_x(identifier)
        y = predict_fn(x)

        # To save disk space
        if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.floating):
            y = y.astype(np.float16)

        np.save(output, y)
        # saving some memory
        del x, y


def evaluate_individual_metrics(load_y_true, metrics: dict, predictions_path, results_path):
    assert len(metrics) > 0, 'No metric provided'

    os.makedirs(results_path)

    results = defaultdict(dict)

    for filename in tqdm(sorted(os.listdir(predictions_path))):
        identifier = np_filename2id(filename)
        y_prob = np.load(os.path.join(predictions_path, filename))
        y_true = load_y_true(identifier)

        for metric_name, metric in metrics.items():
            score = metric(y_true, y_prob)
            if hasattr(score, 'tolist'):
                score = score.tolist()
            results[metric_name][identifier] = score

    for metric_name, result in results.items():
        with open(os.path.join(results_path, metric_name + '.json'), 'w') as f:
            json.dump(result, f, indent=0)


# TODO: deprecated
# Deprecated
# ----------

def find_dice_threshold(load_msegm, ids, predictions_path, thresholds_path):
    """
    Find thresholds for the predicted probabilities that maximize the mean dice score.
    The thresholds are calculated channelwise.

    Parameters
    ----------
    load_msegm: callable(id)
        loader for the multimodal segmentation
    ids: Sequence
        object ids
    predictions_path: str
        path for predicted masks
    thresholds_path: str
        path to store the thresholds
    """
    thresholds = np.linspace(0, 1, 20)
    dices = []

    for patient_id in ids:
        y_true = load_msegm(patient_id)
        y_pred = np.load(os.path.join(predictions_path, f'{patient_id}.npy'))

        # get dice with individual threshold for each channel
        channels = []
        for y_pred_chan, y_true_chan in zip(y_pred, y_true):
            channels.append([dice(y_pred_chan > thr, y_true_chan) for thr in thresholds])
        dices.append(channels)
        # saving some memory
        del y_pred, y_true

    optimal_thresholds = thresholds[np.mean(dices, axis=0).argmax(axis=1)]
    with open(thresholds_path, 'w') as file:
        json.dump(optimal_thresholds.tolist(), file)


def _evaluate(load_y, input_path, output_path, ids, metrics):
    if not metrics:
        return

    os.makedirs(output_path)

    def load_prediction(identifier):
        return np.load(os.path.join(input_path, f'{identifier}.npy'))

    ys, predictions = [], []
    for y, prediction in load_by_ids(load_y, load_prediction, ids=ids):
        ys.append(y)
        predictions.append(prediction)

    result = evaluate_fn(ys, predictions, metrics)

    for name, value in result.items():
        metric = os.path.join(output_path, name)
        if isinstance(value, np.ndarray):
            value = value.tolist()

        with open(metric, 'w') as f:
            json.dump(value, f, indent=2)
