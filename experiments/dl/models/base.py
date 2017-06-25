from abc import ABC, abstractmethod

import tensorflow as tf

from ..optimizer import Optimizer


class Model(ABC):
    def __init__(self, optimizer: Optimizer):
        self.global_step = 0
        self.optimizer = optimizer
        self.graph = tf.get_default_graph()

        self.saver = self.init_op = None
        # Gets initialized during build
        self.train_input_phs = self.inference_input_phs = None
        self.training_ph = None
        self.loss = self.y_pred = None
        self.train_op = self.train_summary_op = None

        # Gets initialized during model preparation
        self.file_writer = None
        self.call_train = self.call_val = self.call_pred = None

    def finalize_build(self):
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.graph.finalize()

    def prepare(self, session: tf.Session, file_writer: tf.summary.FileWriter):
        self.file_writer = file_writer

        self.call_train = session.make_callable(
            [self.train_op, self.loss, self.train_summary_op],
            [*self.train_input_phs, self.optimizer.lr, self.training_ph])

        self.call_val = session.make_callable(
            [self.y_pred, self.loss],
            [*self.train_input_phs, self.training_ph])

        self.call_pred = session.make_callable(
            self.y_pred, [*self.inference_input_phs])

    def do_train_step(self, *train_inputs, lr):
        _, loss, summary = self.call_train(*train_inputs, lr, True)
        self.file_writer.add_summary(summary)

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
