import tensorflow as tf

from dpipe.model_core import ModelCore


class FrozenModel:
    def __init__(self, model_core: ModelCore, predict: callable):
        self.model_core = model_core

        self._build(predict)

        # Gets initialized during model preparation
        self.session = self.file_writer = None
        self.call_train = self.call_val = self.call_pred = None

    def _build(self, predict):
        self.graph = tf.get_default_graph()

        self.x_phs, logits = self.model_core.build(False)
        self.y_pred = predict(logits)

        self.saver = tf.train.Saver()
        self.graph.finalize()

    def prepare(self, session: tf.Session, restore_ckpt_path):
        self.call_pred = session.make_callable(self.y_pred, self.x_phs)
        self.saver.restore(session, restore_ckpt_path)

    def do_inf_step(self, *inference_inputs):
        return self.call_pred(*inference_inputs, False)

    def predict_object(self, x):
        return self.model_core.predict_object(x, self.do_inf_step)
