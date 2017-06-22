import datetime
from os.path import join

import numpy as np
import tensorflow as tf

from experiments.experiments.models import SegmentationModel
from . import summaries
from .optimizer import Optimizer


class BasicModelController:
    def __init__(self, model: SegmentationModel, log_path: str,
                 restore_ckpt_path: str=None):
        self.graph = model.graph

        with self.graph.as_default():

            self.optimization = Optimizer(model.loss)

            log_path = join(log_path, timestamp())
            log_writer = tf.summary.FileWriter(log_path, self.graph, 10, 30)
            self.train_iter_writer = summaries.Scalar(
                {'loss': model.loss, 'lr': self.optimization.lr},
                'train_summary', log_writer)
            self.train_writer = summaries.Custom('avg_losses/train', log_writer)
            self.val_writer = summaries.Custom('avg_losses/val', log_writer)

        self.session = tf.Session(graph=self.graph)
        if restore_ckpt_path is None:
            self.session.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(self.session, restore_ckpt_path)

        self.call_train = self.session.make_callable(
            [self.optimization.train_op, model.loss,
             self.train_iter_writer.summary_op],
            [*model.x_phs, *model.y_phs, self.optimization.lr,
             model.training_ph]
        )
        self.call_val = self.session.make_callable(
            [model.y_pred, model.loss],
            [*model.x_phs, *model.y_phs, model.training_ph]
        )
        self.call_pred_proba = self.session.make_callable(
            model.y_pred_proba,
            [*model.x_phs, model.training_ph]
        )

    def train_iterate(self, inputs, lr):
        _, loss, summary = self.call_train(*inputs, lr, True)

        self.train_iter_writer.write_summary(summary)
        return loss

    def train(self, batch_iter, lr, n_iter: int=None):
        losses = []
        for i, inputs in enumerate(batch_iter):
            loss = self.train_iterate(inputs, lr=lr)
            losses.append(loss)
            if n_iter is not None and i >= n_iter:
                break

        loss = np.mean(losses)
        self.train_writer.write_summary(loss)
        return loss

    def validate(self, inputs, n_iter: int=None):
        y_pred = []
        losses = []
        for i, inputo in enumerate(inputs):
            input = [a[None, :] for a in inputo]
            y_pred_batch, loss = self.call_val(*input, False)
            y_pred.extend(y_pred_batch)
            losses.append(loss)

            if n_iter is not None and i >= n_iter:
                break

        loss = np.mean(losses)
        self.val_writer.write_summary(loss)
        return y_pred, loss

    def predict_proba(self, xos):
        input = [a[None, :] for a in xos]
        y_pred_proba = self.call_pred_proba(*input, False)

        return y_pred_proba[0]


def timestamp():
    d = datetime.datetime.now()
    return d.strftime("%Y/%m/%d/%X")
