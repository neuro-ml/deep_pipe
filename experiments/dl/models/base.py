from abc import ABC, abstractmethod

import tensorflow as tf

from ..optimizer import Optimizer


class Model(ABC):
    def __init__(self, optimizer: Optimizer, n_chans_in, n_chans_out):
        self.optimizer = optimizer
        self.n_chans_in = n_chans_in
        self.n_chans_out = n_chans_out

        # Gets initialized during _build_model
        self.graph = None
        self.train_input_phs = self.inference_input_phs = None
        self.loss = self.y_pred = None
        self.train_op = self.train_summary_op = None

        self._build()

        # Gets initialized during model preparation
        self.global_step = None
        self.file_writer = None
        self.call_train = self.call_val = self.call_pred = None

    def _start_build(self):
        self.graph = tf.get_default_graph()
        self.training_ph = tf.placeholder('bool', name='is_training')

    @abstractmethod
    def _build_model(self):
        """Method defines placeholders and tensors, necessary for the model."""
        pass

    def _build(self):
        self._start_build()
        self._build_model()
        self._finalize_build()

    def _finalize_build(self):
        self.train_op = self.optimizer.build_train_op(self.loss)

        with tf.name_scope('train_summary'):
            tf.summary.scalar('lr', self.optimizer.lr)
            tf.summary.scalar('loss', self.loss)

            self.train_summary_op = tf.summary.merge_all()

        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.graph.finalize()

    def prepare(self, session: tf.Session, file_writer: tf.summary.FileWriter,
                restore_ckpt_path=None):
        self.global_step = 0
        self.file_writer = file_writer

        self.call_train = session.make_callable(
            [self.train_op, self.loss, self.train_summary_op],
            [*self.train_input_phs, self.optimizer.lr, self.training_ph])

        self.call_val = session.make_callable(
            [self.y_pred, self.loss],
            [*self.train_input_phs, self.training_ph])

        self.call_pred = session.make_callable(
            self.y_pred, [*self.inference_input_phs, self.training_ph])

        if restore_ckpt_path is None:
            session.run(self.init_op)
        else:
            self.saver.restore(session, restore_ckpt_path)

    def do_train_step(self, *train_inputs, lr):
        _, loss, summary = self.call_train(*train_inputs, lr, True)
        self.file_writer.add_summary(summary, self.global_step)
        self.global_step += 1

        return loss

    def do_val_step(self, *val_inputs):
        return self.call_val(*val_inputs, False)

    def do_inf_step(self, *inference_inputs):
        return self.call_pred(*inference_inputs, False)

    @abstractmethod
    def validate_object(self, x, y):
        pass

    @abstractmethod
    def predict_object(self, x):
        pass
