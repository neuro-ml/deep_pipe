import os

import tensorflow as tf

from dpipe.config import register
from dpipe.model_core import ModelCore


def get_model_path(save_path):
    return os.path.join(save_path, 'model')


@register('model', 'model')
class Model:
    def __init__(self, model_core: ModelCore, predict: callable, loss: callable,
                 optimize: callable):
        self.model_core = model_core

        self._build(predict, loss, optimize)

    def _build(self, predict, loss, optimize):
        self.graph = tf.get_default_graph()

        training_ph = tf.placeholder('bool', name='is_training')
        x_phs, logits = self.model_core.build(training_ph)

        y_pred = predict(logits)

        lr = tf.placeholder(tf.float32, name='learning_rate')
        loss, y_ph = loss(logits=logits)
        train_op = optimize(loss=loss, lr=lr)

        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.graph.finalize()

        # ----------------------------------------------------------------------
        self.session = tf.Session()

        self.call_train = self.session.make_callable(
            [train_op, loss], [*x_phs, y_ph, lr, training_ph]
        )

        self.call_val = self.session.make_callable(
            [y_pred, loss], [*x_phs, y_ph, training_ph]
        )

        self.call_pred = self.session.make_callable(
            y_pred, [*x_phs, training_ph]
        )

        self.session.run(init_op)

    def do_train_step(self, *train_inputs, lr):
        _, loss = self.call_train(*train_inputs, lr, True)
        return loss

    def do_val_step(self, *val_inputs):
        return self.call_val(*val_inputs, False)

    def do_inf_step(self, *inference_inputs):
        return self.call_pred(*inference_inputs, False)

    def save(self, save_path):
        self.saver.save(self.session, get_model_path(save_path))

    def load(self, load_path):
        self.saver.restore(self.session, get_model_path(load_path))


@register('frozen_model', 'model')
class FrozenModel:
    def __init__(self, model_core: ModelCore, predict: callable,
                 restore_ckpt_path):
        self.model_core = model_core

        self._build(predict, restore_ckpt_path)

    def _build(self, predict, restore_ckpt_path):
        self.graph = tf.get_default_graph()

        x_phs, logits = self.model_core.build(False)
        y_pred = predict(logits)

        self.saver = tf.train.Saver()
        self.graph.finalize()

        self.session = tf.Session()
        self.call_pred = self.session.make_callable(y_pred, [*x_phs])
        self.saver.restore(self.session, get_model_path(restore_ckpt_path))

    def do_inf_step(self, *inference_inputs):
        return self.call_pred(*inference_inputs)
