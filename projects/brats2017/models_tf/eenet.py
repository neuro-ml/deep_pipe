import tensorflow as tf
import tensorflow.contrib.slim as slim

from .utils import batch_norm


def cb(t, n_chans, kernel_size, scope, training, relu=False):
    with tf.variable_scope(scope):
        t = tf.layers.conv3d(t, n_chans, kernel_size, use_bias=False,
                             data_format='channels_first')
        t = batch_norm(t, training=training)

        return t if not relu else tf.nn.relu(t)


def bottleneck(t, int_chans, out_chans, kernel_size, training, scope):
    with tf.variable_scope(scope):
        t1 = cb(t, int_chans, 1, relu=True, training=training, scope='encode')
        t1 = cb(t1, int_chans, kernel_size, relu=True, training=training,
                scope='process')
        t1 = cb(t1, out_chans, 1, training=training, scope='decode')

        # Extract subset of spatial data (center)
        s = kernel_size // 2
        t2 = t[:, :, s:-s, s:-s, s:-s]

        # if in_chans != out_chans
        if t.get_shape()[1] != out_chans:
            t2 = cb(t2, out_chans, 1, scope='transform', training=training)

        return tf.nn.relu(t1 + t2)


class Model:
    def __init__(self, blocks, n_classes, kernel_size):
        self.x_ph = tf.placeholder(tf.float32,
                                   (None, blocks[0], None, None, None),
                                   name='x')
        self.y_ph = tf.placeholder(tf.int64,
                                   (None, None, None, None),
                                   name='y_true')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        t = self.x_ph
        with tf.variable_scope('model'):
            for i, n_chans in enumerate(blocks[1:]):
                t = bottleneck(t, n_chans // 4, n_chans, kernel_size,
                               training=self.is_training,
                               scope='bottleneck_{}'.format(i))

            t = cb(t, n_classes, 1, scope='predict', training=self.is_training)

        self.logits = t
        self.y_pred = tf.nn.softmax(self.logits, 1)

        with tf.name_scope('loss'):
            self.loss = tf.losses.sparse_softmax_cross_entropy(
                self.y_ph, tf.transpose(self.logits, [0, 2, 3, 4, 1]))

        with tf.variable_scope('optimization'):
            opt = tf.train.MomentumOptimizer(self.lr, 0.9, use_nesterov=True)
            self.train_op = slim.learning.create_train_op(self.loss, opt)

        # Code to use tensorboard
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    @property
    def graph(self):
        return tf.get_default_graph()
