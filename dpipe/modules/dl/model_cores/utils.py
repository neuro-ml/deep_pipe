import warnings

import tensorflow as tf
import tensorflow.contrib.slim as slim


def batch_norm(t, momentum=0.9, axis=1, center=True, scale=True,
               training=False):
    assert axis == 1

    warnings.warn("batch_norm is deprecated, use spacial_batch_norm instead",
                  DeprecationWarning)

    return spacial_batch_norm(t, momentum, center, scale, training,
                              data_format="channels_first")


def spacial_batch_norm(t, momentum=0.9, center=True, scale=True, training=False,
                       data_format='channels_last', name='spacial_batch_norm'):
    with tf.variable_scope(name):
        shape = tf.shape(t)

        if data_format == 'channels_first':
            n_chans = t.get_shape().as_list()[1]

            t = tf.reshape(t, (shape[0], n_chans, 1, -1))
            t = slim.batch_norm(t, scale=scale, center=center, decay=momentum,
                                data_format='NCHW', fused=True,
                                is_training=training,
                                scope='fused_batcn_norm_slim')

        elif data_format == 'channels_last':
            n_chans = t.get_shape().as_list()[-1]

            t = tf.reshape(t, (shape[0], 1, -1, n_chans))
            t = slim.batch_norm(t, scale=scale, center=center, decay=momentum,
                                data_format='NHWC', fused=True,
                                is_training=training,
                                scope='fused_batcn_norm_slim')
        else:
            raise ValueError('wrong data_format')

        return tf.reshape(t, shape)
