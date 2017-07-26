import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .base import ModelCore

from dpipe.modules.batch_iterators.slices import \
    iterate_multiple_slices as iterate_slices


def init_block(inputs, name, training, output_channels, kernel_size=3):
    with tf.variable_scope(name):
        inp_channels = int(inputs.shape[1])
        conv = tf.layers.conv2d(inputs, output_channels - inp_channels,
                                kernel_size, strides=2, padding='same',
                                use_bias=False, data_format='channels_first')
        pool = tf.layers.max_pooling2d(inputs, pool_size=2, strides=2,
                                       padding='same',
                                       data_format='channels_first')

        inputs = tf.concat([conv, pool], 1)
        inputs = slim.batch_norm(inputs, decay=0.9, scale=True,
                                 is_training=training,
                                 data_format='NCHW', fused=True)
        return tf.nn.relu(inputs)


def conv_block(inputs, channels, kernel_size, strides, training, padding='same',
               activation=tf.nn.relu, layer=tf.layers.conv2d, name=None):
    conv = layer(inputs, channels, kernel_size=kernel_size, strides=strides,
                 padding=padding, use_bias=False, data_format='channels_first',
                 name=name)
    conv = slim.batch_norm(conv, decay=0.9, scale=True, is_training=training,
                           data_format='NCHW', fused=True)
    return activation(conv)


def res_block(inputs, name, output_channels, training, kernel_size=3,
              downsample=False,
              upsample=False, dropout_prob=.1, internal_scale=4):
    # it can be either upsampling or downsampling:
    assert not (upsample and downsample)

    input_channels = int(inputs.shape[1])
    internal_channels = output_channels // internal_scale
    input_stride = downsample and 2 or 1

    with tf.variable_scope(name):
        # conv path
        # TODO: use prelu
        conv = conv_block(inputs, internal_channels, kernel_size=input_stride,
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
        main = inputs
        if downsample:
            main = tf.layers.max_pooling2d(
                inputs, pool_size=2, strides=2,
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


def stage(inputs, output_channels, num_blocks, name, training,
          downsample=False, upsample=False):
    with tf.variable_scope(name):
        inputs = res_block(inputs, 'res0', output_channels, training=training,
                           downsample=downsample, upsample=upsample)
        for i in range(num_blocks - 1):
            inputs = res_block(inputs, f'res%d' % (i + 1), output_channels,
                               training=training)
        return inputs


def build_model(inputs, classes, name, training, init_channels):
    with tf.variable_scope(name):
        initial_shape = tf.shape(inputs)
        inputs = init_block(inputs, 'init', training, init_channels)
        inputs = stage(inputs, 64, 5, 'stage1', training, downsample=True)
        inputs = stage(inputs, 128, 9, 'stage2', training, downsample=True)
        inputs = stage(inputs, 128, 8, 'stage3', training)
        inputs = stage(inputs, 64, 3, 'stage4', training, upsample=True)
        inputs = stage(inputs, 16, 2, 'stage5', training, upsample=True)
        inputs = tf.layers.conv2d_transpose(
            inputs, classes, kernel_size=2, strides=2, use_bias=True,
            data_format='channels_first')
        inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
        inputs = tf.image.resize_images(inputs, initial_shape[-2:])
        inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
        return inputs


class ENet2D(ModelCore):
    def __init__(self, n_chans_in, n_chans_out, multiplier=1, init_channels=16):
        super().__init__(n_chans_in * multiplier, n_chans_out)
        self.init_channels = init_channels
        self.multiplier = multiplier

    def build(self, training_ph):
        x_ph = tf.placeholder(
            tf.float32, (None, self.n_chans_in, None, None), name='input'
        )

        logits = build_model(x_ph, self.n_chans_out, 'enet_2d',
                             training_ph, self.init_channels)
        return [x_ph], logits

    def validate_object(self, x, y, do_val_step):
        # TODO: add batches
        predicted, losses, weights = [], [], []
        for x_slice, y_slice in iterate_slices(x, y, self.multiplier):
            y_pred, loss = do_val_step(x_slice[None], y_slice[None])

            predicted.extend(y_pred)
            losses.append(loss)
            weights.append(y_pred.size)

        loss = np.average(losses, weights=weights)
        return np.stack(predicted, axis=-1), loss

    def predict_object(self, x, do_inf_step):
        predicted = []
        for x_slice in iterate_slices(x, num_slices=self.multiplier):
            y_pred = do_inf_step(x_slice[None])
            predicted.extend(y_pred)

        return np.stack(predicted, axis=-1)
