from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim

softmax = partial(tf.nn.softmax, dim=1)
sigmoid = tf.nn.sigmoid


def optimize(*, loss, lr, tf_optimizer_name, **params):
    with tf.variable_scope('optimization'):
        optimizer = getattr(tf.train, tf_optimizer_name)(
            lr, name='optimizer', **params)
        return slim.learning.create_train_op(loss, optimizer)


def sparse_softmax_cross_entropy(*, logits, y_ph):
    with tf.variable_scope('sparse_softmax_cross_entropy'):
        t = tf.transpose(logits, [0, *range(2, len(logits.shape)), 1])
        return tf.losses.sparse_softmax_cross_entropy(labels=y_ph, logits=t)


def sigmoid_cross_entropy(*, logits, y_ph):
    return tf.losses.sigmoid_cross_entropy(y_ph, logits=logits)
