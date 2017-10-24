import os
import json

import numpy as np
from tqdm import tqdm

from dpipe.config import register
from dpipe.medim.metrics import dice_score as dice
from dpipe.medim.metrics import multichannel_dice_score


@register()
def train_model(train_fn, model, save_model_path, restore_model_path):
    if restore_model_path is not None:
        model.load(restore_model_path)

    train_fn()
    model.save(save_model_path)


@register()
def transform(input_path, output_path, transform_fn):
    os.makedirs(output_path)

    for f in tqdm(os.listdir(input_path)):
        np.save(os.path.join(output_path, f), transform_fn(np.load(os.path.join(input_path, f))))


@register()
def predict(ids, output_path, load_x, predict_object):
    os.makedirs(output_path)

    for identifier in tqdm(ids):
        x = load_x(identifier)
        y = predict_object(x)

        np.save(os.path.join(output_path, str(identifier)), y)
        # saving some memory
        del x, y


@register()
def compute_dices(load_msegm, predictions_path, dices_path):
    dices = {}
    for f in tqdm(os.listdir(predictions_path)):
        patient_id = f.replace('.npy', '')
        y_true = load_msegm(patient_id)
        y = np.load(os.path.join(predictions_path, f))

        dices[patient_id] = multichannel_dice_score(y, y_true)

    with open(dices_path, 'w') as f:
        json.dump(dices, f, indent=0)


@register()
def find_dice_threshold(load_msegm, predictions_path, thresholds_path):
    thresholds = np.linspace(0, 1, 20)
    dices = []

    for f in tqdm(os.listdir(predictions_path)):
        threshold_ids = f.replace('.npy', '')
        y_true = load_msegm(threshold_ids)
        y_pred = np.load(os.path.join(predictions_path, f))

        # get dice with individual threshold for each channel
        for y_pred_chan, y_true_chan in zip(y_pred, y_true):
            dices.append([dice(y_pred_chan > thr, y_true_chan) for thr in thresholds])

        # saving some memory
        del y_pred, y_true
    optimal_thresholds = thresholds[np.mean(dices, axis=0).argmax(axis=1)]
    np.save(thresholds_path, optimal_thresholds)
