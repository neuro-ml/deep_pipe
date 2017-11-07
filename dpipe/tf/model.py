import tensorflow as tf

from dpipe.config import register
from dpipe.model import Model, FrozenModel, get_model_path
from dpipe.model_core import ModelCore


@register('tf', 'model')
class TFModel(Model):
    def __init__(self, model_core: ModelCore, logits2pred: callable, logits2loss: callable, optimize: callable):
        self.model_core = model_core

        self._build(logits2pred, logits2loss, optimize)

    def _build(self, logits2pred, logits2loss, optimize):
        self.graph = tf.get_default_graph()

        training_ph = tf.placeholder('bool', name='is_training')
        x_phs, logits = self.model_core.build(training_ph)

        y_pred = logits2pred(logits)

        lr = tf.placeholder(tf.float32, name='learning_rate')
        loss, y_ph = logits2loss(logits=logits)
        train_op = optimize(loss=loss, lr=lr)

        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.graph.finalize()

        # ----------------------------------------------------------------------
        self.session = tf.Session()

        self.call_train = self.session.make_callable([train_op, loss],
                                                     [*x_phs, y_ph, lr, training_ph])
        self.call_val = self.session.make_callable([y_pred, loss],
                                                   [*x_phs, y_ph, training_ph])
        self.call_pred = self.session.make_callable(y_pred,
                                                    [*x_phs, training_ph])

        self.session.run(init_op)

    def do_train_step(self, *train_inputs, lr):
        _, loss = self.call_train(*train_inputs, lr, True)
        return loss

    def do_val_step(self, *val_inputs):
        return self.call_val(*val_inputs, False)

    def do_inf_step(self, *inference_inputs):
        return self.call_pred(*inference_inputs, False)

    def save(self, path):
        self.saver.save(self.session, get_model_path(path))

    def load(self, path):
        self.saver.restore(self.session, get_model_path(path))


@register('tf', 'frozen_model')
class TFFrozenModel(FrozenModel):
    def __init__(self, model_core: ModelCore, logits2pred: callable, restore_model_path):
        self.model_core = model_core

        self._build(logits2pred, restore_model_path)

    def _build(self, logits2pred, restore_model_path):
        self.graph = tf.get_default_graph()

        x_phs, logits = self.model_core.build(False)
        y_pred = logits2pred(logits)

        self.saver = tf.train.Saver()
        self.graph.finalize()

        self.session = tf.Session()
        self.call_pred = self.session.make_callable(y_pred, [*x_phs])
        self.saver.restore(self.session, get_model_path(restore_model_path))

    def do_inf_step(self, *inference_inputs):
        return self.call_pred(*inference_inputs)
