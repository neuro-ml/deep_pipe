import numpy as np
import tensorflow as tf

import medim

from .base import ModelCore
from .utils import batch_norm


def activation(t):
    return tf.nn.relu(t)


def bac(t, n_chans, kernel_size, training, name):
    with tf.variable_scope(name):
        t = batch_norm(t, training=training)
        t = activation(t)
        t = tf.layers.conv3d(
            t, n_chans, kernel_size, data_format='channels_first',
            use_bias=False)
        return t


# Residual Block
def res_block(t, n_chans, kernel_size, training, name):
    s = kernel_size - 1

    with tf.variable_scope(name):
        with tf.variable_scope('transform'):
            t2 = t
            if s > 0:
                t2 = t[:, :, s:-s, s:-s, s:-s]

            n_chans_dif = n_chans - tf.shape(t2)[1]
            if n_chans_dif != 0:
                t2 = tf.pad(t2, [(0, 0), (0, n_chans_dif)] + [(0, 0)] * 3)

        t1 = t
        t1 = bac(t1, n_chans, kernel_size, training, 'block_a')
        t1 = bac(t1, n_chans, kernel_size, training, 'block_b')

        t3 = t1 + t2

    return t3


def build_path(t, blocks, kernel_size, training, name):
    with tf.variable_scope(name):
        t = bac(t, blocks[0], kernel_size, training, 'BAC_0')
        t = bac(t, blocks[0], kernel_size, training, 'BAC_1')

        for i, n_chans in enumerate(blocks[1:]):
            t = res_block(t, n_chans, kernel_size, training, f'ResBlock_{i}')

        return t


def build_model(t_det, t_context, kernel_size, n_classes, training, name,
                path_blocks=(30, 40, 40, 50), n_chans_common=150):
    with tf.variable_scope(name):
        t_det = build_path(t_det, path_blocks, kernel_size, training,
                           'detailed')

        t_context = tf.layers.average_pooling3d(
            t_context, 3, 3, 'same', 'channels_first')

        t_context = build_path(t_context, path_blocks, kernel_size, training,
                               'context')

        with tf.variable_scope('upconv'):
            t_context_up = batch_norm(t_context, training=training)
            t_context_up = activation(t_context_up)
            t_context_up = tf.layers.conv3d_transpose(
                t_context_up, path_blocks[-1], kernel_size, strides=[3, 3, 3],
                data_format='channels_first', use_bias=False)

        t_comm = tf.concat([t_context_up, t_det], axis=1)
        t_comm = res_block(t_comm, n_chans_common, 1, training,
                           name='comm')

        t = bac(t_comm, n_classes, 1, training, 'C')
        t = batch_norm(t, training=training)
        logits = t

        return logits


class DeepMedic(ModelCore):
    def __init__(self, *, n_chans_in, n_chans_out, n_parts):
        super().__init__(n_chans_in=n_chans_in, n_chans_out=n_chans_out)

        self.kernel_size = 3
        self.n_parts = np.array(n_parts)
        assert np.all((self.n_parts == 1) | (self.n_parts == 2))

    def build(self, training_ph):
        nan = None
        x_det_ph = tf.placeholder(
            tf.float32, (nan, self.n_chans_in, nan, nan, nan), name='x_det')
        x_con_ph = tf.placeholder(
            tf.float32, (nan, self.n_chans_in, nan, nan, nan), name='x_con')
        y_ph = tf.placeholder(
            tf.float32, (nan, self.n_chans_out, nan, nan, nan), name='y_true')

        self.train_input_phs = [x_det_ph, x_con_ph, y_ph]
        self.inference_input_phs = [x_det_ph, x_con_ph]

        logits = build_model(
            x_det_ph, x_con_ph, self.kernel_size, self.n_chans_out,
            training_ph, name='deep_medic')

        self.y_pred = tf.nn.sigmoid(logits, name='y_pred')

        self.loss = tf.losses.log_loss(y_ph, self.y_pred, scope='loss')

    def validate_object(self, x, y, do_val_step):
        x_shape = np.array(x.shape)
        x_det_padding, x_con_padding, y_padding = find_padding(
            x_shape, self.n_parts)

        x_det_parts, x_con_parts = prepare_x(x, x_det_padding, x_con_padding,
                                             self.n_parts)
        y_parts = prepare_y(y, y_padding, self.n_parts)

        y_pred_parts, weights, losses = [], [], []
        for x_det, x_con, y_part in zip(x_det_parts, x_con_parts, y_parts):
            y_pred, loss = do_val_step(x_det[None, :], x_con[None, :],
                                       y_part[None, :])
            y_pred_parts.append(y_pred[0])
            losses.append(loss)
            weights.append(y_pred.size)

        loss = np.average(losses, weights=weights)
        y_pred = medim.split.combine(y_pred_parts, [1, *self.n_parts])
        y_pred = restore_y(y_pred, y_padding, self.n_parts)
        return y_pred, loss

    def predict_object(self, x, do_inf_step):
        x_shape = np.array(x.shape)
        x_det_padding, x_con_padding, y_padding = find_padding(
            x_shape, self.n_parts)

        x_det_parts, x_con_parts = prepare_x(x, x_det_padding, x_con_padding,
                                             self.n_parts)

        y_pred_parts = []
        for x_det, x_con in zip(x_det_parts, x_con_parts):
            y_pred = do_inf_step(x_det[None, :], x_con[None, :])[0]
            y_pred_parts.append(y_pred)

        y_pred = medim.split.combine(y_pred_parts, [1, *self.n_parts])
        y_pred = restore_y(y_pred, y_padding, self.n_parts)
        return y_pred


def find_padding(x_shape, n_parts):
    quants = n_parts * 3

    y_padding = np.zeros((4, 2), dtype=int)
    y_padding[1:, 1] = (quants - x_shape[1:] % quants) % quants

    x_det_padding = y_padding.copy()
    x_det_padding[1:] += 8

    x_con_padding = y_padding.copy()
    x_con_padding[1:] += 24

    return x_det_padding, x_con_padding, y_padding


def prepare_x(x, x_det_padding, x_con_padding, n_parts):
    x_det = np.pad(x, x_det_padding, mode='constant')
    x_con = np.pad(x, x_con_padding, mode='constant')

    x_det_parts = medim.split.divide(x_det, [0, 8, 8, 8],
                                     n_parts_per_axis=[1, *n_parts])

    x_con_parts = medim.split.divide(x_con, [0, 24, 24, 24],
                                     n_parts_per_axis=[1, *n_parts])
    return x_det_parts, x_con_parts


def prepare_y(y, y_padding, n_parts):
    y = np.pad(y, y_padding, mode='constant')
    y_parts = medim.split.divide(y, [0] * 4,
                                 n_parts_per_axis=[1, *n_parts])

    return y_parts


def restore_y(y_pred, y_padding, n_parts):
    r_border = np.array(y_pred.shape[1:]) - y_padding[1:, 1]
    return y_pred[:, :r_border[0], :r_border[1], :r_border[2]]
