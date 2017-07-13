import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .base import ModelCore


def iterate_slices(*data, axis=-1, empty=True):
    """
    Iterate over slices of a series of tensors along a given axis.
    If empty is False, the last tensor in the series is assumed to be a mask.

    Parameters
    ----------
    data: list, tuple, np.array
    axis: int
    empty: bool
        whether to yield slices, containing only zeroes in the mask.
    """
    for i in range(data[0].shape[axis]):
        if empty or data[-1].take(i, axis=axis).any():
            result = [entry.take(i, axis=axis) for entry in data]
            if len(data) == 1:
                result = result[0]
            yield result


def init_block(input, name, training, kernel_size=3):
    with tf.variable_scope(name):
        inp_channels = int(input.shape[1])
        conv = tf.layers.conv2d(input, 16 - inp_channels, kernel_size,
                                strides=2, padding='same',
                                use_bias=False, data_format='channels_first')
        pool = tf.layers.max_pooling2d(input, pool_size=2, strides=2,
                                       padding='same',
                                       data_format='channels_first')

        input = tf.concat([conv, pool], 1)
        input = slim.batch_norm(input, decay=0.9, scale=True,
                                is_training=training,
                                data_format='NCHW', fused=True)
        return tf.nn.relu(input)


def conv_block(input, channels, kernel_size, strides, training, padding='same',
               activation=tf.nn.relu, layer=tf.layers.conv2d, name=None):
    conv = layer(input, channels, kernel_size=kernel_size, strides=strides,
                 padding=padding, use_bias=False, data_format='channels_first',
                 name=name)
    conv = slim.batch_norm(conv, decay=0.9, scale=True, is_training=training,
                           data_format='NCHW', fused=True)
    return activation(conv)


def res_block(input, name, output_channels, training, kernel_size=3,
              downsample=False,
              upsample=False, dropout_prob=.1, internal_scale=4):
    # it can be either upsampling or downsampling:
    assert not (upsample and downsample)

    input_channels = int(input.shape[1])
    internal_channels = output_channels // internal_scale
    input_stride = downsample and 2 or 1

    with tf.variable_scope(name):
        # conv path
        # TODO: use prelu
        conv = conv_block(input, internal_channels, kernel_size=input_stride,
                          strides=input_stride, training=training, name='conv1')

        if upsample:
            conv = conv_block(conv, internal_channels, kernel_size,
                              strides=2, training=training,
                              layer=tf.layers.conv2d_transpose, name='upsample')
        else:
            # TODO: use dilated and asymmetric convolutions
            conv = conv_block(conv, internal_channels, kernel_size, strides=1,
                              training=training, name='no_upsample')
        conv = conv_block(conv, output_channels, kernel_size=1, strides=1,
                          training=training, name='conv3')
        conv = tf.layers.dropout(conv, dropout_prob, training=training)

        # main path
        main = input
        if downsample:
            main = tf.layers.max_pooling2d(
                input, pool_size=2, strides=2,
                padding='same', data_format='channels_first')
        if output_channels != input_channels:
            main = conv_block(main, output_channels, kernel_size=1, strides=1,
                              activation=tf.identity, training=training,
                              name='justify')
        if upsample:
            main = tf.layers.conv2d_transpose(
                main, output_channels, kernel_size, strides=2, padding='same',
                use_bias=False, data_format='channels_first',
                name='justify_upsample')
        return tf.nn.relu(conv + main)


def stage(input, output_channels, num_blocks, name, training,
          downsample=False, upsample=False):
    with tf.variable_scope(name):
        input = res_block(input, 'res0', output_channels, training=training,
                          downsample=downsample, upsample=upsample)
        for i in range(num_blocks - 1):
            input = res_block(input, f'res%d' % (i + 1), output_channels,
                              training=training)
        return input


def build_model(input, classes, name, training):
    with tf.variable_scope(name):
        initial_shape = tf.shape(input)
        input = init_block(input, 'init', training)
        input = stage(input, 64, 5, 'stage1', training, downsample=True)
        input = stage(input, 128, 9, 'stage2', training, downsample=True)
        input = stage(input, 128, 8, 'stage3', training)
        input = stage(input, 64, 3, 'stage4', training, upsample=True)
        input = stage(input, 16, 2, 'stage5', training, upsample=False)
        input = tf.layers.conv2d_transpose(
            input, classes, kernel_size=2, strides=2, use_bias=True,
            data_format='channels_first')
        input = tf.transpose(input, perm=[0, 2, 3, 1])
        input = tf.image.resize_images(input, initial_shape[-2:])
        input = tf.transpose(input, perm=[0, 3, 1, 2])
        return input


class ENet2D(ModelCore):
    def build(self, training_ph):
        x_ph = tf.placeholder(
            tf.float32, (None, self.n_chans_in, None, None), name='input'
        )
        y_ph = tf.placeholder(
            tf.float32, (None, self.n_chans_out, None, None), name='y_true'
        )

        self.train_input_phs = [x_ph, y_ph]
        self.inference_input_phs = [x_ph]

        model = build_model(x_ph, self.n_chans_out, 'enet_2d', training_ph)
        self.y_pred = tf.nn.sigmoid(model, name='y_pred')

        self.loss = tf.losses.log_loss(y_ph, self.y_pred, scope='loss')

    def validate_object(self, x, y, do_val_step):
        predicted, losses, weights = [], [], []
        for x_slice, y_slice in iterate_slices(x, y, empty=True):
            y_pred, loss = do_val_step(x_slice[None], y_slice[None])

            predicted.extend(y_pred)
            losses.append(loss)
            weights.append(y_pred.size)

        loss = np.average(losses, weights=weights)
        return np.stack(predicted, axis=-1), loss

    def predict_object(self, x, do_inf_step):
        predicted, losses, = [], []
        for x_slice in iterate_slices(x, empty=True):
            y_pred = do_inf_step(x_slice[None])
            predicted.extend(y_pred)

        return np.stack(predicted, axis=-1)
