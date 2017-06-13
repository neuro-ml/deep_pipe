import gc
from tqdm import tqdm

import numpy as np

import tensorflow as tf

import tfmod

from utils import compute_dices_msegm, make_check_decrease, get_dice_threshold

batch_size = 64
n_batches_per_epoch = 100
n_epoch = 150

patience = 5
rtol = 0.004
atol = 0.004

lr_dec = 0.8


def extract(l, idx):
    return [l[i] for i in idx]


def train_model(model_controller, make_batch_iterator: callable,
                make_val_inputs, data_loader, train_idx, val_idx, n_epoch):
    def compute_dices(y_pred, y_true):
        dices = []
        for yo_pred, yo_true in zip(y_pred, y_true):
            msegm_true = data_loader.segm2msegm(yo_true)
            msegm_pred = data_loader.segm2msegm(yo_pred)
            dices.append(compute_dices_msegm(msegm_pred, msegm_true))
        return np.mean(dices, axis=0)

    patients = data_loader.patients

    x_train, x_val = [], []
    y_train, y_val = [], []

    print('Loading train data', flush=True)
    for patient in tqdm(extract(patients, train_idx)):
        x_train.append(data_loader.load_mscan(patient))
        y_train.append(data_loader.load_segm(patient))

    print('Loading val data', flush=True)
    for patient in tqdm(extract(patients, val_idx)):
        x_val.append(data_loader.load_mscan(patient))
        y_val.append(data_loader.load_segm(patient))

    val_inputs = make_val_inputs(x_val, y_val)

    # Start training
    print('Starting training', flush=True)
    lr = 0.1
    check_should_decrease_lr = make_check_decrease(patience, rtol, atol)

    train_iter = make_batch_iterator(x_train, y_train, batch_size=batch_size)

    for epoch in range(n_epoch):
        gc.collect()
        print('Epoch {}'.format(epoch), flush=True)

        train_loss = model_controller.train(train_iter, lr, n_batches_per_epoch)
        print('Train:', train_loss, flush=True)

        y_pred, val_loss = model_controller.validate(val_inputs)
        print('Val  :', val_loss)

        #val_dices = compute_dices(y_pred, (i[-1] for i in val_inputs))
        #print('Val dices :', val_dices)

        print('\n', flush=True)

        if check_should_decrease_lr(train_loss):
            check_should_decrease_lr = make_check_decrease(patience, rtol, atol)
            lr *= lr_dec


def find_threshold(model_controller, get_pred_and_true, sum_probs,
                   data_loader, val_idx):
    def compute_dices(y_pred, y_true):
        dices = []
        for yo_pred, yo_true in zip(y_pred, y_true):
            msegm_true = data_loader.segm2msegm(yo_true)
            msegm_pred = data_loader.segm2msegm(yo_pred)
            dices.append(compute_dices_msegm(msegm_pred, msegm_true))
        return np.mean(dices, axis=0)

    patients = data_loader.patients

    x_val, y_val = [], []

    print('Loading val data', flush=True)
    for patient in tqdm(extract(patients, val_idx)):
        x_val.append(data_loader.load_mscan(patient))
        y_val.append(data_loader.load_segm(patient))


    print('Starting prediction', flush=True)
    y_pred, y_true = [], []
    for xo_val, yo_val in zip(x_val, y_val):
        yo_pred, yo_true = get_pred_and_true(model_controller, xo_val, yo_val)
        y_pred.append(yo_pred)
        y_true.append(yo_true)

    msegms_pred = [sum_probs(y) for y in y_pred]
    msegms_true = [data_loader.segm2msegm(y) for y in y_true]

    thresholds = get_dice_threshold(msegms_pred, msegms_true)
    return np.array(thresholds)


def get_stats_and_dices(model_controller, get_pred_and_true, sum_probs,
                        data_loader, data_idx, threshold):
    def compute_dices(y_pred, y_true):
        dices = []
        for yo_pred, yo_true in zip(y_pred, y_true):
            msegm_true = data_loader.segm2msegm(yo_true)
            msegm_pred = data_loader.segm2msegm(yo_pred)
            dices.append(compute_dices_msegm(msegm_pred, msegm_true))
        return np.mean(dices, axis=0)

    patients = data_loader.patients

    x, y = [], []

    print('Loading test data', flush=True)
    for patient in tqdm(extract(patients, data_idx)):
        x.append(data_loader.load_mscan(patient))
        y.append(data_loader.load_segm(patient))


    print('Starting prediction', flush=True)
    dices = []
    stats_pred, stats_true = [], []
    for xo_val, yo_val in tqdm(zip(x, y)):
        yo_pred, yo_true = get_pred_and_true(model_controller, xo_val, yo_val)
        msegm_pred = sum_probs(yo_pred) > threshold[:, None, None, None]
        msegm_true = data_loader.segm2msegm(yo_true)

        stats_pred.append(msegm_pred.sum(axis=(1, 2, 3)))
        stats_true.append(msegm_true.sum(axis=(1, 2, 3)))

        dices.append(compute_dices_msegm(msegm_pred, msegm_true))

    return np.array(stats_pred), np.array(stats_true), np.array(dices)
