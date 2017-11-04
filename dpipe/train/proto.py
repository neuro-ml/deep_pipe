import math
from collections import defaultdict
from functools import partial

import numpy as np
from tensorboard_easy.logger import Logger

from dpipe.batch_iter_factory import BatchIterFactory
from dpipe.batch_predict import BatchPredict
from dpipe.config import register
from dpipe.dl.model import Model
from .utils import make_find_next_lr, make_check_loss_decrease


@register()
def proto(model: Model, train_batch_iter_factory: BatchIterFactory, batch_predict: BatchPredict, n_epochs,
          log_path, lr_init, lr_dec_mul, patience, rtol, atol,
          load_x, load_y, val_ids, single=None, multiple=None):
    # lr policy
    # stopping policy
    logger = Logger(log_path)

    xs_val = [load_x(val_id) for val_id in val_ids]
    ys_val = [load_y(val_id) for val_id in val_ids]

    find_next_lr = make_find_next_lr(lr_init, lambda lr: lr * lr_dec_mul,
                                     partial(make_check_loss_decrease, patience=patience, rtol=rtol, atol=atol))

    train_log_write = logger.make_log_scalar('train/loss')
    lr_log_write = logger.make_log_scalar('train/lr')
    train_avg_log_write = logger.make_log_scalar('epoch/train_loss')
    val_avg_log_write = logger.make_log_scalar('epoch/val_loss')

    lr = find_next_lr(math.inf)
    with train_batch_iter_factory, logger:
        for epoch in range(n_epochs):
            with next(train_batch_iter_factory) as train_batch_iter:
                train_losses = []
                for inputs in train_batch_iter:
                    train_losses.append(model.do_train_step(*inputs, lr=lr))
                    train_log_write(train_losses[-1])
                train_avg_log_write(np.mean(train_losses))

            metrics = defaultdict(list)
            val_predictions = []
            val_losses = []

            for x, y in zip(xs_val, ys_val):
                y_pred, loss = batch_predict.validate(x, y, validate_fn=model.do_val_step)
                val_losses.append(loss)
                val_predictions.append(y_pred)

                if single:
                    for name, metric in single.items():
                        metrics[name].append(metric(y, y_pred))

            # TODO: move asarray to tb-easy
            for name, values in metrics.items():
                logger.log_histogram(f'metrics/{name}', np.asarray(values), epoch)

            if multiple:
                for name, metric in multiple.items():
                    value = metric(ys_val, val_predictions)
                    logger.log_scalar(f'metrics/{name}', value, epoch)

            val_loss = np.mean(val_losses)
            val_avg_log_write(val_loss)
            lr = find_next_lr(val_loss)
            lr_log_write(lr)
