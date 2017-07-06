import unittest

import numpy as np

from .deepmedic_orig import DeepMedic
from ..optimizer import Optimizer
import tensorflow as tf


class TestDeepMedic(unittest.TestCase):
    def setUp(self):
        optimizer = Optimizer()
        self.n_chans_in = 4
        self.n_chans_out = 3
        self.n_parts = np.array([1, 2, 1])
        self.model = DeepMedic(optimizer, self.n_chans_in, self.n_chans_out,
                               n_parts=self.n_parts)

        self.session = tf.Session()
        self.session.run(self.model.init_op)

        self.writer = tf.summary.FileWriter('/tmp/')
        self.model.prepare(self.session, self.writer)

        x_det_shape = np.array([self.n_chans_in, 25, 25, 25])
        self.x_det = np.arange(np.product(x_det_shape), dtype=np.float32)\
            .reshape(x_det_shape)

        x_con_shape = np.array([self.n_chans_in, 57, 57, 57])
        self.x_con = np.arange(np.product(x_con_shape), dtype=np.float32)\
            .reshape(x_con_shape)

        y_shape = np.array([self.n_chans_out, 9, 9, 9])
        self.y = np.ones(np.product(y_shape), dtype=np.float32) \
            .reshape(y_shape)

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_train(self):
        loss = self.model.do_train_step(
            self.x_det[None, :], self.x_con[None, :], self.y[None, :], lr=0.1)
        self.assertFalse(np.isnan(loss))

    def test_val(self):
        y_pred, loss = self.model.do_val_step(
            self.x_det[None, :], self.x_con[None, :], self.y[None, :])
        self.assertFalse(np.isnan(loss))

    def test_inference_behaviour(self):
        class FakeModel(DeepMedic):
            def __init__(self, n_parts):
                self.n_parts = n_parts
                pass

            def do_inf_step(self, *inference_inputs):
                return inference_inputs[0][:, 8:-8, 8:-8, 8:-8]

        model = FakeModel(self.n_parts)

        for i in range(5):
            shape = [self.n_chans_in] + list(np.random.randint(50, 180, size=3))
            input = np.arange(np.prod(shape)).reshape(shape)
            output = model.predict_object(input)

            self.assertSequenceEqual(output.shape, input.shape)
            self.assertSequenceEqual(list(input.flatten()),
                                     list(output.flatten()))
