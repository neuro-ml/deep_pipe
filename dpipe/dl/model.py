import tensorflow as tf

from dpipe.model_cores import ModelCore
from .summaries import SummaryLogger


class Model:
    def __init__(self, model_core: ModelCore, predict: callable, loss: callable,
                 optimize: callable):
        self.model_core = model_core

        self._build(predict, loss, optimize)

        # Gets initialized during model preparation
        self.session = self.file_writer = None
        self.call_train = self.call_val = self.call_pred = None

    def _build(self, predict, loss, optimize):
        self.graph = tf.get_default_graph()

        self.training_ph = tf.placeholder('bool', name='is_training')
        self.x_phs, logits = self.model_core.build(self.training_ph)

        self.y_pred = predict(logits)

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.y_ph = tf.placeholder(tf.float32, logits.shape, name='y_true')
        self.loss = loss(logits=logits, y_ph=self.y_ph)
        self.train_op = optimize(loss=self.loss, lr=self.lr)

        self.summary_logger = SummaryLogger(
            'training_summary', {'lr': self.lr, 'loss': self.loss})

        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.graph.finalize()

    def prepare(self, session: tf.Session, file_writer: tf.summary.FileWriter,
                restore_ckpt_path=None):
        self.session = session
        self.file_writer = file_writer

        self.call_train = session.make_callable(
            [self.train_op, self.loss, self.summary_logger.summary_op],
            [*self.x_phs, self.y_ph, self.lr, self.training_ph])

        self.call_val = session.make_callable(
            [self.y_pred, self.loss],
            [*self.x_phs, self.y_ph, self.training_ph])

        self.call_pred = session.make_callable(
            self.y_pred, [*self.x_phs, self.training_ph])

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
