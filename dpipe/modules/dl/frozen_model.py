import tensorflow as tf

from .model_cores import ModelCore


class FrozenModel:
    def __init__(self, model_core: ModelCore):
        self.model_core = model_core
        self._build()

        # Gets initialized during model preparation
        self.session = self.call_pred = None

    def _start_build(self):
        self.graph = tf.get_default_graph()

    def _build(self):
        self._start_build()
        self.model_core.build(False)
        self._finalize_build()

    def _finalize_build(self):
        self.saver = tf.train.Saver()
        self.graph.finalize()

    def prepare(self, session: tf.Session, restore_ckpt_path):
        self.session = session

        self.call_pred = session.make_callable(
            self.model_core.y_pred, [*self.model_core.inference_input_phs])

        self.saver.restore(session, restore_ckpt_path)

    def do_inf_step(self, *inference_inputs):
        return self.call_pred(*inference_inputs)

    def predict_object(self, x):
        return self.model_core.predict_object(x, self.do_inf_step)
