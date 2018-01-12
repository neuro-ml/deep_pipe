import tensorflow as tf

from .base import ModelCore
from .layers import spatial_batch_norm

from dpipe.config import register


def activation(t):
    return tf.nn.relu(t)


def bac(t, n_chans, kernel_size, training, name):
    with tf.variable_scope(name):
        t = spatial_batch_norm(t, training=training,
                               data_format='channels_first')
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
            t_context_up = spatial_batch_norm(t_context, training=training,
                                              data_format='channels_first')
            t_context_up = activation(t_context_up)
            t_context_up = tf.layers.conv3d_transpose(
                t_context_up, path_blocks[-1], kernel_size, strides=[3, 3, 3],
                data_format='channels_first', use_bias=False)

        t_comm = tf.concat([t_context_up, t_det], axis=1)
        t_comm = res_block(t_comm, n_chans_common, 1, training,
                           name='comm')

        t = bac(t_comm, n_classes, 1, training, 'C')
        t = spatial_batch_norm(t, training=training,
                               data_format='channels_first')
        logits = t

        return logits


class DeepMedicOrig(ModelCore):
    def __init__(self, *, n_chans_in, n_chans_out):
        super().__init__(n_chans_in=n_chans_in, n_chans_out=n_chans_out)

        self.kernel_size = 3

    def build(self, training_ph):
        nan = None
        x_det_ph = tf.placeholder(
            tf.float32, (nan, self.n_chans_in, nan, nan, nan), name='x_det')
        x_con_ph = tf.placeholder(
            tf.float32, (nan, self.n_chans_in, nan, nan, nan), name='x_con')

        logits = build_model(
            x_det_ph, x_con_ph, self.kernel_size, self.n_chans_out,
            training_ph, name='deep_medic')

        return [x_det_ph, x_con_ph], logits
