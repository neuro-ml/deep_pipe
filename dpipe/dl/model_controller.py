import numpy as np
import tensorflow as tf

from dpipe.config import register
from .summaries import make_write_value
from .model import Model


@register('model_controller', 'model_controller')
class ModelController:
    def __init__(self, model: Model, log_path, restore_model_path=None,
                 metrics: dict = None):
        self.model = model
        self.log_path = log_path
        self.restore_model_path = restore_model_path
        self.metrics = metrics
        if metrics is None:
            self.metrics = {}

    def _start(self):
        self.session = tf.Session(graph=self.model.graph)

        self.file_writer = tf.summary.FileWriter(
            self.log_path, self.model.graph, 10, 30)
        self.write_avg_loss_train = make_write_value(
            'epoch_stats/train_loss', self.file_writer)
        self.write_avg_loss_val = make_write_value(
            'epoch_stats/val_loss', self.file_writer)
        self.write_metrics = {
            k: make_write_value('epoch_stats/' + k, self.file_writer)
            for k in self.metrics}

        self.model.prepare(self.session, self.file_writer,
                           restore_ckpt_path=self.restore_model_path)

    def train(self, batch_iter, *, lr):
        losses = [self.model.do_train_step(*inputs, lr=lr)
                  for inputs in batch_iter]
        loss = np.mean(losses)
        self.write_avg_loss_train(loss)
        return loss

    def validate(self, xs, ys):
        ys_pred = []
        losses = []
        for x, y in zip(xs, ys):
            y_pred, loss = self.model.validate_object(x, y)
            ys_pred.append(y_pred)
            losses.append(loss)

        for k, compute_metric in self.metrics.items():
            self.write_metrics[k](compute_metric(y_true=ys, y_pred=ys_pred))

        loss = np.mean(losses)
        self.write_avg_loss_val(loss)
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
