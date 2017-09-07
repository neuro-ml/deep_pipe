import functools

import numpy as np
import tensorflow as tf

from dpipe import medim
from .base import ModelCore
from .utils import spatial_batch_norm, prelu

activation = functools.partial(prelu, feature_dims=[1])


def cb(t, n_chans, kernel_size, training, name):
    with tf.variable_scope(name):
        t = tf.layers.conv3d(t, n_chans, kernel_size, use_bias=False,
                             data_format='channels_first')
        t = spatial_batch_norm(t, training=training,
                               data_format='channels_first')
        return t


def cba(t, n_chans, kernel_size, training, name):
    with tf.variable_scope(name):
        return activation(cb(t, n_chans, kernel_size, training, 'cb'))


def build_path(t, blocks, kernel_size, training, name):
    with tf.variable_scope(name):
        for i, n_chans in enumerate(blocks):
            t = cba(t, n_chans, kernel_size, training, f'cba_{i}_a')
            t = cba(t, n_chans, kernel_size, training, f'cba_{i}_b')

        return t


downsampling_ops = {
    'average': lambda x: tf.layers.average_pooling3d(x, 3, 3, 'same',
                                                     'channels_first'),
    'sampling': lambda x: x[:, :, 1:-1:3, 1:-1:3, 1:-1:3]
}


def build_model(t_det_in, t_con_in, kernel_size, n_classes, training, name, *,
                path_blocks=(30, 40, 40, 50), n_chans_com=150, dropout):
    with tf.variable_scope(name):
        t_det = build_path(t_det_in, path_blocks, kernel_size, training,
                           'detailed')

        t_con = build_path(t_con_in, path_blocks, kernel_size, training,
                           'context')

        t_con_up = t_con
        with tf.variable_scope('upconv'):
            t_con_up = tf.layers.conv3d_transpose(
                t_con_up, path_blocks[-1], 3, strides=[3, 3, 3],
                data_format='channels_first', use_bias=False)
            t_con_up = spatial_batch_norm(t_con_up, training=training,
                                          data_format='channels_first')
            t_con_up = activation(t_con_up)

        t_com = tf.concat([t_con_up, t_det], axis=1)

        t_com = cba(dropout(t_com), n_chans_com, 1, training, name='comm_1')
        t_com = cba(dropout(t_com), n_chans_com, 1, training, name='comm_2')
        return cb(t_com, n_classes, 1, training, 'C')


class DeepMedicEls(ModelCore):
    def __init__(self, *, n_chans_in, n_chans_out, n_parts,
                 downsampling_type='sampling', with_dropout=True):
        super().__init__(n_chans_in=n_chans_in, n_chans_out=n_chans_out)

        self.kernel_size = 3
        self.with_dropout = with_dropout
        self.downsampling_op = downsampling_ops[downsampling_type]
        self.n_parts = np.array(n_parts)
        assert np.all(np.in1d(self.n_parts, [1, 2]))

    def build(self, training_ph):
        if self.with_dropout:
            dropout = lambda x: tf.layers.dropout(
                x, training=training_ph,
                noise_shape=(tf.shape(x)[0], tf.shape(x)[1], 1, 1, 1),
            )
        else:
            dropout = lambda x: x

        input_shape = (None, self.n_chans_in, None, None, None)
        x_det_ph = tf.placeholder(tf.float32, input_shape, name='x_det')
        x_con_ph = tf.placeholder(tf.float32, input_shape, name='x_con')

        x_con = self.downsampling_op(x_con_ph)

        logits = build_model(x_det_ph, x_con, self.kernel_size,
                             self.n_chans_out, training_ph, name='deep_medic',
                             dropout=dropout)

        return [x_det_ph, x_con_ph], logits

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
        y_pred = restore_y(y_pred, y_padding)
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
        y_pred = restore_y(y_pred, y_padding)
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
    y = np.pad(y, y_padding[-y.ndim:], mode='constant')
    n_parts_per_axis = [1] * (y.ndim - 3) + list(n_parts)
    y_parts = medim.split.divide(y, [0] * y.ndim,
                                 n_parts_per_axis=n_parts_per_axis)

    return y_parts


def restore_y(y_pred, y_padding):
    r_border = np.array(y_pred.shape[1:]) - y_padding[1:, 1]
    return y_pred[:, :r_border[0], :r_border[1], :r_border[2]]
