import math
from functools import partial

import numpy as np

from dpipe.dl.model_controller import ModelController
from dpipe.utils.batch_iter_factory import BatchIterFactory
from dpipe.medim.metrics import multichannel_dice_score
from .utils import make_find_next_lr, make_check_loss_decrease
from dpipe.config import register


@register()
def train_segm(
        model_controller: ModelController,
        train_batch_iter_factory: BatchIterFactory,
        val_ids, dataset, *, n_epochs, lr_init, lr_dec_mul=0.5,
        patience: int, rtol=0, atol=0):
    val_x = [dataset.load_mscan(p) for p in val_ids]
    val_segm = [dataset.load_segm(p) for p in val_ids]
    val_msegm = [dataset.load_msegm(p) for p in val_ids]

    find_next_lr = make_find_next_lr(
        lr_init, lambda lr: lr * lr_dec_mul,
        partial(make_check_loss_decrease, patience=patience,
                rtol=rtol, atol=atol))

    lr = find_next_lr(math.inf)
    with train_batch_iter_factory:
        for i in range(n_epochs):
            with next(train_batch_iter_factory) as train_batch_iter:
                train_loss = model_controller.train(train_batch_iter, lr=lr)

            y_pred_proba, val_loss = model_controller.validate(val_x, val_segm)

            y_pred = [np.argmax(y, axis=0) for y in y_pred_proba]
            msegm_pred = [dataset.segm2msegm(y) for y in y_pred]

            dices = [multichannel_dice_score(pred, true)
                     for pred, true in zip(msegm_pred, val_msegm)]

            print('{:>5} {:>10.5f} {}'.format(i, val_loss,
                                              np.mean(dices, axis=0)))

            lr = find_next_lr(val_loss)
