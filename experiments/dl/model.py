import tensorflow as tf

from .model_cores import ModelCore
from .optimizer import Optimizer
from .summaries import SummaryLogger


class Model:
    def __init__(self, model_core: ModelCore, optimizer: Optimizer):
        self.model_core = model_core
        self.optimizer = optimizer
        self._build()

        # Gets initialized during model preparation
        self.session = self.file_writer = None
        self.call_train = self.call_val = self.call_pred = None

    def _start_build(self):
        self.graph = tf.get_default_graph()
        self.training_ph = tf.placeholder('bool', name='is_training')

    def _build(self):
        self._start_build()
        self.model_core.build(self.training_ph)
        self._finalize_build()

    def _finalize_build(self):
        self.train_op = self.optimizer.build_train_op(self.model_core.loss)

        self.summary_logger = SummaryLogger('training_summary',
                                            {'lr': self.optimizer.lr,
                                             'loss': self.model_core.loss})

        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.graph.finalize()

    def prepare(self, session: tf.Session, file_writer: tf.summary.FileWriter,
                restore_ckpt_path=None):
        self.session = session
        self.file_writer = file_writer

        self.call_train = session.make_callable(
            [self.train_op, self.model_core.loss,
             self.summary_logger.summary_op],
            [*self.model_core.train_input_phs, self.optimizer.lr,
             self.training_ph])

        self.call_val = session.make_callable(
            [self.model_core.y_pred, self.model_core.loss],
            [*self.model_core.train_input_phs, self.training_ph])

        self.call_pred = session.make_callable(
            self.model_core.y_pred, [*self.model_core.inference_input_phs,
                                     self.training_ph])

        if restore_ckpt_path is None:
            session.run(self.init_op)
        else:
            self.saver.restore(session, restore_ckpt_path)

    def do_train_step(self, *train_inputs, lr):
        _, loss, summary = self.call_train(*train_inputs, lr, True)
        self.summary_logger.write(summary, self.file_writer)

        return loss

    def do_val_step(self, *val_inputs):
        return self.call_val(*val_inputs, False)

    def do_inf_step(self, *inference_inputs):
        return self.call_pred(*inference_inputs, False)

    def validate_object(self, x, y):
        return self.model_core.validate_object(x, y, self.do_val_step)

    def predict_object(self, x):
        return self.model_core.predict_object(x, self.do_inf_step)

    def save(self, save_path):
        self.saver.save(self.session, save_path)
