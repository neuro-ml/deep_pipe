import math
from functools import partial

import numpy as np
from tensorboard_easy.logger import Logger

from dpipe.batch_iter_factory import BatchIterFactory
from dpipe.batch_predict import BatchPredict
from dpipe.config import register
from dpipe.dl.model import Model
from dpipe.medim.metrics import multichannel_dice_score
from .logging import make_log_vector
from .utils import make_find_next_lr, make_check_loss_decrease


@register()
def train_segm(model: Model, train_batch_iter_factory: BatchIterFactory, batch_predict: BatchPredict, log_path, val_ids,
               dataset, *, n_epochs, lr_init, lr_dec_mul=0.5, patience: int, rtol=0, atol=0):
    logger = Logger(log_path)

    mscans_val = [dataset.load_mscan(p) for p in val_ids]
    segms_val = [dataset.load_segm(p) for p in val_ids]
    msegms_val = [dataset.load_msegm(p) for p in val_ids]

    find_next_lr = make_find_next_lr(lr_init, lambda lr: lr * lr_dec_mul,
                                     partial(make_check_loss_decrease, patience=patience, rtol=rtol, atol=atol))

    train_log_write = logger.make_log_scalar('train_loss')
    train_avg_log_write = logger.make_log_scalar('avg_train_loss')
    val_avg_log_write = logger.make_log_scalar('avg_val_loss')
    val_dices_log_write = make_log_vector(logger, 'val_dices')

    lr = find_next_lr(math.inf)
    with train_batch_iter_factory, logger:
        for i in range(n_epochs):
            with next(train_batch_iter_factory) as train_batch_iter:
                train_losses = []
                for inputs in train_batch_iter:
                    train_losses.append(model.do_train_step(*inputs, lr=lr))
                    train_log_write(train_losses[-1])
                train_avg_log_write(np.mean(train_losses))

            msegms_pred = []
            val_losses = []
            for x, y in zip(mscans_val, segms_val):
                y_pred, loss = batch_predict.validate(x, y, validate_fn=model.do_val_step)
                msegms_pred.append(dataset.segm2msegm(np.argmax(y_pred, axis=0)))
                val_losses.append(loss)

            val_loss = np.mean(val_losses)
            val_avg_log_write(val_loss)
            val_dices = [multichannel_dice_score(pred, true) for pred, true in zip(msegms_pred, msegms_val)]
            val_dices_log_write(np.mean(val_dices, axis=0))

            lr = find_next_lr(val_loss)
