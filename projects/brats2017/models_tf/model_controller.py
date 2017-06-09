import datetime
from os.path import join

import tensorflow as tf


def make_training_operation(model, log_path):
    writer = build_file_writer('train', model, log_path)

    def train(x_batch, y_batch, lr, session):
        _, loss, summary = session.run(
            [model.train_op, model.loss, model.summary_op],
            {model.x_ph: x_batch, model.y_ph: y_batch,
             model.lr: lr, model.is_training: True})
        writer.add_summary(summary)
        return loss
    return train


def make_validation_operation(model, log_path):
    writer = build_file_writer('val', model, log_path)

    def validate(x_batch, y_batch, session):
        y_pred, loss, summary = session.run(
            [model.y_pred, model.loss, model.summary_op],
            {model.x_ph: x_batch, model.y_ph: y_batch,
             model.is_training: False})
        writer.add_summary(summary)
        return y_pred, loss
    return validate


def build_file_writer(name, model, log_path):
    log_path = join(log_path, timestamp(), name)
    return tf.summary.FileWriter(log_path, model.graph)


def timestamp():
    d = datetime.datetime.now()
    return d.strftime("%Y/%m/%d/%X")
