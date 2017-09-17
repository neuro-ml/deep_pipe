from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim

from dpipe.config import register


@register(module_type='predict')
def softmax(logits):
    with tf.variable_scope('softmax'):
        return tf.nn.softmax(logits=logits, dim=1, )


sigmoid = register(module_type='predict')(
    partial(tf.nn.sigmoid, name='sigmoid')
)


@register('tf_optimize', 'optimize')
def optimize(loss, lr, *, tf_optimizer_name, **params):
    with tf.variable_scope('optimization'):
        optimizer = getattr(tf.train, tf_optimizer_name)(
            lr, name='optimizer', **params)
        return slim.learning.create_train_op(loss, optimizer)


@register(module_type='loss')
def sparse_softmax_cross_entropy(*, logits):
    with tf.variable_scope('sparse_softmax_cross_entropy'):
        y_ph_shape = logits.shape[0:1].concatenate(logits.shape[2:])
        y_ph = tf.placeholder(tf.int32, y_ph_shape, name='y_true')

        t = tf.transpose(logits, [0, *range(2, len(logits.shape)), 1])
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y_ph, logits=t)
    return loss, y_ph


@register(module_type='loss')
def sigmoid_cross_entropy(*, logits):
    y_ph = tf.placeholder(tf.float32, logits.shape, name='y_true')
    return tf.losses.sigmoid_cross_entropy(y_ph, logits=logits), y_ph


@register(module_type='loss')
def soft_dice_loss(*, logits, y_ph, softness=1e-7):
    batch = tf.shape(y_ph)[0]

    y_ph = tf.reshape(y_ph, [batch, -1])
    logits = tf.reshape(logits, [batch, -1])

    intersection = 2 * tf.reduce_sum(logits * y_ph, axis=1)
    volumes = tf.reduce_sum(logits, axis=1) + tf.reduce_sum(y_ph, axis=1)

    return - tf.reduce_mean((intersection + softness) / (volumes + softness))
