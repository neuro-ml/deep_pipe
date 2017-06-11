import datetime

import tensorflow as tf
import tensorflow.contrib.slim as slim


def batch_norm(t, momentum=0.9, axis=1, center=True, scale=True,
               training=False):
    assert axis == 1

    with tf.variable_scope('batch_norm'):
        shape = tf.shape(t)
        n_chans = t.get_shape().as_list()[1]

        t = tf.reshape(t, (shape[0], n_chans, 1, -1))
        t = slim.batch_norm(t, scale=scale, center=center, decay=momentum,
                            data_format='NCHW', fused=True,
                            is_training=training,
                            scope='fused_batcn_norm_slim')
        t = tf.reshape(t, shape)
        return t
