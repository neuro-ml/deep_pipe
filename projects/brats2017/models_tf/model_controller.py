import datetime
from os.path import join

import numpy as np
import tensorflow as tf

from . import modules


class ModelController:
    def __init__(self, model, log_path, restore_ckpt_path=None):
        self.graph = model.graph

        with self.graph.as_default():

            self.optimization = modules.Optimizer(model.loss)

            log_path = join(log_path, timestamp())
            log_writer = tf.summary.FileWriter(log_path, self.graph,
                                               flush_secs=30)
            self.train_writer = modules.summaries.Scalar(
                {'loss': model.loss, 'lr': self.optimization.lr},
                'train_summary', log_writer)
            self.val_writer = modules.summaries.Custom(
                'mean_val_loss', log_writer)

        self.session = tf.Session(graph=self.graph)
        if restore_ckpt_path is None:
            self.session.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(self.session, restore_ckpt_path)

        self.call_train = self.session.make_callable(
            [self.optimization.train_op, model.loss,
             self.train_writer.summary_op],
            [*model.x_phs, model.y_ph, self.optimization.lr, model.is_training]
        )
        self.call_val = self.session.make_callable(
            [model.y_pred, model.loss],
            [*model.x_phs, model.y_ph, model.is_training]
        )
        self.call_predict = self.session.make_callable(
            model.y_pred,
            [*model.x_phs, model.is_training]
        )
        self.call_predict_proba = self.session.make_callable(
            model.y_pred_proba,
            [*model.x_phs, model.is_training]
        )

    def train_iterate(self, x_batches, y_batch, lr):
        _, loss, summary = self.call_train(*x_batches, y_batch, lr, True)

        self.train_writer.write_summary(summary)
        return loss

    def train(self, batch_iter, lr, n_iter):
        losses = []
        for i in range(n_iter):
            *x_batches, y_batch = next(batch_iter)
            loss = self.train_iterate(x_batches, y_batch, lr=lr)

            losses.append(loss)
        return np.mean(losses)

    def validate(self, xs, y):
        y_pred_all = []
        losses = []
        for *xos, yo in zip(*xs, y):
            y_pred, loss = self.call_val(
                *[xo[None, :] for xo in xos], yo[None, :], False)
            y_pred_all.append(y_pred[0])
            losses.append(loss)

        loss = np.mean(losses)

        self.val_writer.write_summary(loss)
        return y_pred_all, loss

    def predict(self, xs):
        y_pred = []
        for xos in zip(*xs):
            yo_pred = self.call_predict(*[xo[None, :] for xo in xos], False)[0]
            y_pred.append(yo_pred)
        return y_pred

    def predict_proba(self, xs):
        y_pred_proba = []
        for xos in zip(*xs):
            yo_pred_proba = \
            self.call_predict_proba(*[xo[None, :] for xo in xos], False)[0]
            y_pred_proba.append(yo_pred_proba)
        return y_pred_proba


def timestamp():
    d = datetime.datetime.now()
    return d.strftime("%Y/%m/%d/%X")
