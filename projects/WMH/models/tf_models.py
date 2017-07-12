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
                            is_training=training, scope='fused_batcn_norm_slim')
        t = tf.reshape(t, shape)
        return t


def activation(t):
    return tf.nn.relu(t)


class UnetLike:
    def __init__(self):
        self.x_ph = tf.placeholder(tf.float32, (None, 2, None, None, None))
        self.y_ph = tf.placeholder(tf.int32, (None, 1, None, None, None))
        self.training = tf.placeholder(tf.bool)
        self.lr = tf.placeholder(tf.float32)

        self.layers_down = [self.down_conv(16, 3), self.cnn_down_blocks(16, 3),
                            self.down_conv(32, 2, 2),
                            self.cnn_down_blocks(32, 64),
                            self.cnn_down_blocks(56, 64)]

        self.layers_inside = [self.cnn_down_blocks(56, 86), self.up_conv(64, 3)]

        self.layers_up = [self.up_conv(64, 3), self.up_conv(16, 3),
                          self.up_conv(32, 2, 2), self.up_conv(8, 3),
                          self.up_conv(3, 3)]

        self.pred = None
        self.summary = None
        self.loss = None
        self.train_op = None

    def build(self):
        with tf.variable_scope('model'):
            tensors = [self.x_ph]
            for i, apply_layer in enumerate(self.layers_down):
                name_ = 'down_' + str(i + 1)
                tensors.append(apply_layer(tensors[-1], name_, self.training))
            x = tensors[-1]
            for i, apply_layer in enumerate(self.layers_inside):
                name_ = 'inside_' + str(i)
                x = apply_layer(x, name_, self.training)
            for i, (apply_layer, left) in enumerate(
                    zip(self.layers_up, reversed(tensors))):
                name_ = 'up_' + str(i)
                x = tf.concat((left, x), 1)
                x = apply_layer(x, name_, self.training)
            self.pred = x

        self.loss = tf.losses.sparse_softmax_cross_entropy(
            tf.reshape(self.y_ph, [-1, 1]), tf.reshape(self.pred, [-1, 3]))

        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            self.summary = tf.summary.merge_all()
        self.train_op = slim.learning.create_train_op(self.loss,
                                                      tf.train.AdamOptimizer(
                                                          learning_rate=self.lr))

    def validate_object(self, x, y, do_val_step):
        pass

    def predict_object(self, x, do_inf_step):
        pass

    # TODO clarify static or classmethod realization.
    def cnn_down_blocks(self, mid_channels, out_channels):
        """Like a bottleneck from resnet50, but without residual connections."""
        def bottleneck(t, name, training):
            with tf.variable_scope(name):
                channels = [mid_channels, mid_channels, out_channels]
                filters = [1, 3, 1]
                for i, (n_ch, n_filters) in enumerate(zip(channels, filters)):
                    with tf.variable_scope(name + 'down_conv_' + str(i)):
                        t = tf.layers.conv3d(t, n_ch, n_filters,
                                             data_format='channels_first',
                                             use_bias=False)
                        t = activation(batch_norm(t, training=training))
            return t
        return bottleneck

    # TODO clarify static or classmethod realization.
    def down_conv(self, n_channels, filter, strides=1):
        def conv(t, name, training):
            with tf.variable_scope(name):
                t = tf.layers.conv3d(t, n_channels, filter, strides=strides,
                                     data_format='channels_first',
                                     use_bias=False)
                t = batch_norm(t, training=training)
                t = activation(t)
            return t
        return conv

    # TODO clarify static or classmethod realization.
    def up_conv(self, n_channels, filter, strides=1):
        def conv(t, name, training):
            with tf.variable_scope(name):
                t = tf.layers.conv3d_transpose(t, n_channels, filter,
                                               strides=strides,
                                               data_format='channels_first',
                                               use_bias=False)
                # t = batch_norm(t, training=training)
                t = activation(t)
            return t
        return conv

