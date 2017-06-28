import numpy as np
import tensorflow as tf

from .summaries import CustomSummaryWriter
from .models import Model


class ModelController:
    def __init__(self, model: Model, log_path, restore_model_path=None):
        self.model = model
        self.log_path = log_path
        self.restore_model_path = restore_model_path

    def _start(self):
        self.session = tf.Session(graph=self.model.graph)

        self.file_writer = tf.summary.FileWriter(
            self.log_path, self.model.graph, 10, 30)
        self.avg_train_summary = CustomSummaryWriter(
            'avg_losses/train', self.file_writer)
        self.avg_val_summary = CustomSummaryWriter(
            'avg_losses/val', self.file_writer)

        self.model.prepare(self.session, self.file_writer,
                           restore_ckpt_path=self.restore_model_path)

    def train(self, batch_iter, lr, n_iter: int=None):
        losses = []
        for i, inputs in enumerate(batch_iter):
            loss = self.model.do_train_step(*inputs, lr=lr)
            losses.append(loss)
            if n_iter is not None and i >= n_iter:
                break

        loss = np.mean(losses)
        self.avg_train_summary.write(loss)
        return loss

    def validate(self, xs, ys, n_iter: int=None):
        ys_pred = []
        losses = []
        for i, (x, y) in enumerate(zip(xs, ys)):
            y_pred, loss = self.model.validate_object(x, y)
            ys_pred.append(y_pred)
            losses.append(loss)

            if n_iter is not None and i >= n_iter:
                break

        loss = np.mean(losses)
        self.avg_val_summary.write(loss)
        return ys_pred, loss

    def predict_object(self, x):
        return self.model.predict_object(x)

    def _stop(self):
        self.session.close()
        self.file_writer.flush()
        self.file_writer.close()

    def __enter__(self):
        self._start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop()
