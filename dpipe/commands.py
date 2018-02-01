"""
Contains a few more sophisticated commands
that are usually accessed via the `do.py` script.
"""

import os
import json

import numpy as np
from tqdm import tqdm

from dpipe.batch_predict import BatchPredict
from dpipe.medim.metrics import dice_score as dice
from dpipe.medim.metrics import multichannel_dice_score
from dpipe.medim.utils import load_by_ids
from dpipe.model import FrozenModel
from dpipe.train.validator import evaluate as evaluate_fn


def train_model(train, model, save_model_path, restore_model_path=None):
    if restore_model_path is not None:
        model.load(restore_model_path)

    train()
    model.save(save_model_path)


def transform(input_path, output_path, transform_fn):
    os.makedirs(output_path)

    for f in tqdm(os.listdir(input_path)):
        np.save(os.path.join(output_path, f), transform_fn(np.load(os.path.join(input_path, f))))


def predict(ids, output_path, load_x, frozen_model: FrozenModel, batch_predict: BatchPredict):
    os.makedirs(output_path)

    for identifier in tqdm(ids):
        x = load_x(identifier)
        y = batch_predict.predict(x, predict_fn=frozen_model.do_inf_step)

        np.save(os.path.join(output_path, str(identifier)), y)
        # saving some memory
        del x, y


def evaluate(load_y, input_path, output_path, ids, metrics):
    if not metrics:
        return
    
    os.makedirs(output_path)

    def load_prediction(identifier):
        return np.load(os.path.join(input_path, f'{identifier}.npy'))

    ys, predictions = [], []
    for y, prediction in load_by_ids(load_y, load_prediction, ids):
        ys.append(y)
        predictions.append(prediction)

    result = evaluate_fn(ys, predictions, metrics)

    for name, value in result.items():
        metric = os.path.join(output_path, name)
        if type(value) is np.ndarray:
            value = value.tolist()

        with open(metric, 'w') as f:
            json.dump(value, f, indent=2)


def compute_dices(load_msegm, predictions_path, dices_path):
    dices = {}
    for f in tqdm(os.listdir(predictions_path)):
        patient_id = f.replace('.npy', '')
        y_true = load_msegm(patient_id)
        y = np.load(os.path.join(predictions_path, f))

        dices[patient_id] = multichannel_dice_score(y, y_true)

    with open(dices_path, 'w') as f:
        json.dump(dices, f, indent=0)


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
