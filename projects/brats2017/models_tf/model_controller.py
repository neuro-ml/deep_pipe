import datetime
from os.path import join

import numpy as np
import tensorflow as tf


class ModelController:
    def __init__(self, model, log_path, restore_ckpt_path=None):
        log_path = join(log_path, timestamp())
        self.graph = model.graph
        self.train_writer = build_file_writer('train', self.graph, log_path)
        self.val_writer = build_file_writer('val', self.graph, log_path,
                                            max_queue=1)

        self.train_iter = 0
        self.val_iter = 0

        self.session = tf.Session(graph=self.graph)
        if restore_ckpt_path is None:
            self.session.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(self.session, restore_ckpt_path)

        self.call_train = self.session.make_callable(
            [model.train_op, model.loss, model.summary_op],
            [model.x_ph, model.y_ph, model.lr, model.is_training]
        )
        self.call_val = self.session.make_callable(
            [model.y_pred, model.loss],
            [model.x_ph, model.y_ph, model.is_training]
        )
        self.call_predict = self.session.make_callable(
            model.y_pred,
            [model.x_ph, model.is_training]
        )
        self.call_predict_proba = self.session.make_callable(
            model.y_pred_proba,
            [model.x_ph, model.is_training]
        )

    def train_iterate(self, x_batch, y_batch, lr):
        _, loss, summary = self.call_train(x_batch, y_batch, lr, True)

        self.train_writer.add_summary(summary, self.train_iter)
        self.train_iter += 1
        return loss

    def train(self, batch_iter, lr, n_iter):
        losses = []
        for i in range(n_iter):
            x_batch, y_batch = next(batch_iter)
            loss = self.train_iterate(x_batch, y_batch, lr)

            losses.append(loss)
        return np.mean(losses)

    def validate(self, x, y):
        y_pred_all = []
        losses = []
        for xo, yo in zip(x, y):
            y_pred, loss = self.call_val(xo[None, :], yo[None, :], False)
            y_pred_all.append(y_pred[0])
            losses.append(loss)

        loss = np.mean(losses)
        summary = tf.Summary(
            value=[tf.Summary.Value(tag='mean_loss_val',
                                    simple_value=loss)])

        self.val_writer.add_summary(summary, global_step=self.val_iter)
        self.val_iter += 1
        return y_pred_all, loss

    def predict(self, x):
        return [self.call_predict(xo[None, :], False)[0] for xo in x]

    def predict_proba(self, x):
        return [self.call_predict_proba(xo[None, :], False)[0] for xo in x]


def build_file_writer(name, graph, log_path, max_queue=10):
    log_path = join(log_path, name)
    return tf.summary.FileWriter(log_path, graph, max_queue, flush_secs=30)


def timestamp():
    d = datetime.datetime.now()
    return d.strftime("%Y/%m/%d/%X")
