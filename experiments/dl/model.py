import tensorflow as tf

from .model_cores import ModelCore


class Model:
    def __init__(self, model: ModelCore):
        self.model = model
        self._build()

        # Gets initialized during model preparation
        self.global_step = None
        self.file_writer = None
        self.call_train = self.call_val = self.call_pred = None

    def _start_build(self):
        self.graph = tf.get_default_graph()
        self.training_ph = tf.placeholder('bool', name='is_training')

    def _build(self):
        self._start_build()
        self.model.build(self.training_ph)
        self._finalize_build()

    def _finalize_build(self):
        self.train_op = self.model.optimizer.build_train_op(self.model.loss)

        with tf.name_scope('train_summary'):
            tf.summary.scalar('lr', self.model.optimizer.lr)
            tf.summary.scalar('loss', self.model.loss)

            self.train_summary_op = tf.summary.merge_all()

        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.graph.finalize()

    def prepare(self, session: tf.Session, file_writer: tf.summary.FileWriter,
                restore_ckpt_path=None):
        self.global_step = 0
        self.file_writer = file_writer

        self.call_train = session.make_callable(
            [self.train_op, self.model.loss, self.train_summary_op],
            [*self.model.train_input_phs, self.model.optimizer.lr,
             self.training_ph])

        self.call_val = session.make_callable(
            [self.model.y_pred, self.model.loss],
            [*self.model.train_input_phs, self.training_ph])

        self.call_pred = session.make_callable(
            self.model.y_pred, [*self.model.inference_input_phs,
                                self.training_ph])

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

    def validate_object(self, x, y):
        return self.model.validate_object(x, y, self.do_val_step)

    def predict_object(self, x):
        return self.model.predict_object(x, self.do_inf_step)
