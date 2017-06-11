import tensorflow as tf
import tensorflow.contrib.slim as slim

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
            t2 = t[:, :, s:-s, s:-s, s:-s]

            n_chans_dif = n_chans - tf.shape(t)[1]
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
                path_blocks=[30, 40, 40, 50], n_chans_common=150):
    with tf.variable_scope(name):
        t_det = build_path(t_det, path_blocks, kernel_size, training,
                           'detailed')

        t_context = tf.nn.avg_pool3d(
            t_context, [1, 1, 3, 3, 3], [1, 1, 3, 3, 3], 'SAME', 'NCDHW')

        t_context = build_path(t_context, path_blocks, kernel_size, training,
                               'conext')

        with tf.variable_scope('upconv'):
            t_context_up = batch_norm(t_context, training=training)
            t_context_up = activation(t_context_up)
            t_context_up = tf.layers.conv3d_transpose(
                t_context_up, path_blocks[-1], kernel_size, strides=[3, 3, 3],
                data_format='channels_first', use_bias=False)

        t_comm = tf.concat([t_context_up, t_det], axis=1)
        t_comm = res_block(t_comm, n_chans_common, kernel_size, training,
                           name='comm')

        t = bac(t_comm, n_classes, 1, training, 'C')
        t = batch_norm(t, training=training)
        logits = t

        return logits


class DeepMedic:
    def __init__(self, n_chans_in, n_classes):
        self.x_det_ph = tf.placeholder(tf.float32,
                                       (None, n_chans_in, None, None, None),
                                       name='x_det')

        self.x_con_ph = tf.placeholder(tf.float32,
                                       (None, n_chans_in, None, None, None),
                                       name='x_con')

        self.y_ph = tf.placeholder(tf.int64,
                                   (None, None, None, None),
                                   name='y_true')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.logits = build_model(self.x_det_ph, self.x_con_ph, 3, n_classes,
                                  self.is_training, 'deep_medic')

        with tf.name_scope('predict_proba'):
            self.y_pred_proba = tf.nn.softmax(self.logits, 1)

        with tf.name_scope('predict'):
            self.y_pred = tf.argmax(self.logits, axis=1)

        with tf.name_scope('loss'):
            self.loss = tf.losses.sparse_softmax_cross_entropy(
                self.y_ph, tf.transpose(self.logits, [0, 2, 3, 4, 1]))

        self.x_phs = [self.x_det_ph, self.x_con_ph]

    @property
    def graph(self):
        return tf.get_default_graph()
