import tensorflow as tf


def prelu(x, feature_dims):
    with tf.variable_scope('prelu'):
        shape = [s if i in feature_dims else 1
                 for i, s in enumerate(x.get_shape().as_list())]

        alphas = tf.get_variable('alpha', shape, dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
        pos = tf.nn.relu(x)
        neg = alphas * (x - tf.abs(x)) * 0.5

    return pos + neg
