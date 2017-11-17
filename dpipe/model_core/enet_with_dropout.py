import numpy as np
import tensorflow as tf

from dpipe.config import register
from .layers import spatial_batch_norm
from .base import ModelCore

from dpipe.medim.slices import iterate_slices


def init_block(inputs, name, training, output_channels, kernel_size=3,
               conv_layer=tf.layers.conv2d, pool_layer=tf.layers.max_pooling2d):
    with tf.variable_scope(name):
        inp_channels = int(inputs.shape[1])
        conv = conv_layer(inputs, output_channels - inp_channels, kernel_size,
                          strides=2, padding='same', use_bias=False,
                          data_format='channels_first')
        pool = pool_layer(inputs, pool_size=2, strides=2, padding='same',
                          data_format='channels_first')

        inputs = tf.concat([conv, pool], 1)
        inputs = spatial_batch_norm(inputs, training=training,
                                    data_format='channels_first')
        return tf.nn.relu(inputs)


def conv_block(inputs, channels, kernel_size, strides, training, padding='same',
               activation=tf.nn.relu, layer=tf.layers.conv2d, name=None):
    with tf.variable_scope(name):
        inputs = layer(inputs, channels, kernel_size=kernel_size,
                       strides=strides, padding=padding, use_bias=False,
                       data_format='channels_first', name=name)
        inputs = spatial_batch_norm(inputs, training=training,
                                    data_format='channels_first')
        return activation(inputs)


def res_block(inputs, name, output_channels, training, kernel_size=3,
              downsample=False, upsample=False, dropout_prob=.1,
              internal_scale=4, conv_down=tf.layers.conv2d,
              conv_up=tf.layers.conv2d_transpose,
              pool_layer=tf.layers.max_pooling2d):
    # it can be either upsampling or downsampling:
    assert not (upsample and downsample)

    input_channels = int(inputs.shape[1])
    internal_channels = output_channels // internal_scale
    input_stride = downsample and 2 or 1

    with tf.variable_scope(name):
        # conv path
        # TODO: use prelu
        conv = conv_block(inputs, internal_channels, kernel_size=input_stride,
                          strides=input_stride, training=training, name='conv1',
                          layer=conv_down)

        if upsample:
            conv = conv_block(conv, internal_channels, kernel_size,
                              strides=2, training=training,
                              layer=conv_up, name='upsample')
        else:
            # TODO: use dilated and asymmetric convolutions
            conv = conv_block(conv, internal_channels, kernel_size, strides=1,
                              training=training, name='no_upsample',
                              layer=conv_down)
        conv = conv_block(conv, output_channels, kernel_size=1, strides=1,
                          training=training, name='conv3', layer=conv_down)
        conv = tf.layers.dropout(conv, dropout_prob, training=training)

        # main path
        main = inputs
        if downsample:
            main = pool_layer(
                inputs, pool_size=2, strides=2,
                padding='same', data_format='channels_first')
        if output_channels != input_channels:
            main = conv_block(main, output_channels, kernel_size=1, strides=1,
                              activation=tf.identity, training=training,
                              name='justify', layer=conv_down)
        if upsample:
            main = conv_up(
                main, output_channels, kernel_size, strides=2, padding='same',
                use_bias=False, data_format='channels_first',
                name='justify_upsample')
        return tf.nn.relu(conv + main)


def stage(inputs, output_channels, num_blocks, name, training,
          downsample=False, upsample=False, conv_down=tf.layers.conv2d,
          conv_up=tf.layers.conv2d_transpose,
          pool_layer=tf.layers.max_pooling2d):
    layers = dict(conv_down=conv_down, conv_up=conv_up, pool_layer=pool_layer)
    with tf.variable_scope(name):
        inputs = res_block(inputs, 'res0', output_channels, training=training,
                           downsample=downsample, upsample=upsample, **layers)
        for i in range(num_blocks - 1):
            inputs = res_block(inputs, f'res%d' % (i + 1), output_channels,
                               training=training, **layers)
        return inputs


def build_model(inputs, classes, name, training, init_channels,
                conv_down=tf.layers.conv2d, conv_up=tf.layers.conv2d_transpose,
                pool_layer=tf.layers.max_pooling2d):
    layers = dict(conv_down=conv_down, conv_up=conv_up, pool_layer=pool_layer)
    with tf.variable_scope(name):
        input_shape = tf.shape(inputs)[2:]

        inputs = init_block(inputs, 'init', training, init_channels,
                            conv_layer=conv_down, pool_layer=pool_layer)
        inputs = stage(inputs, 64, 5, 'stage1', training, downsample=True,
                       **layers)
        inputs = stage(inputs, 128, 9, 'stage2', training, downsample=True,
                       **layers)
        inputs = stage(inputs, 128, 8, 'stage3', training,
                       **layers)
        inputs = stage(inputs, 64, 3, 'stage4', training, upsample=True,
                       **layers)
        inputs = stage(inputs, 16, 2, 'stage5', training, upsample=True,
                       **layers)

        # magic of dropout is here
        inputs = tf.layers.conv2d(inputs, 80, kernel_size=1,
                                  padding='same', use_bias=False,
                                  data_format='channels_first', name='conv_before_dropout')

        inputs = tf.layers.dropout(inputs, training=True, name='dropout')
        # end of magic

        inputs = conv_up(
            inputs, classes, kernel_size=2, strides=2, use_bias=True,
            data_format='channels_first')

        # crop
        output_shape = tf.shape(inputs)[2:]
        begin = (output_shape - input_shape) // 2
        end = begin + input_shape
        idx = [Ellipsis]
        for i in range(begin.shape[0]):
            idx.append(slice(begin[i], end[i]))
        inputs = inputs[idx]

        return inputs


@register('enet2d_with_dropout')
class ENet2D_with_dropout(ModelCore):
    def __init__(self, n_chans_in, n_chans_out, multiplier=1, init_channels=16):
        super().__init__(n_chans_in * multiplier, n_chans_out)
        self.init_channels = init_channels
        self.multiplier = multiplier

    def build(self, training_ph):
        x_ph = tf.placeholder(
            tf.float32, (None, self.n_chans_in, None, None), name='input'
        )

        logits = build_model(x_ph, self.n_chans_out, 'enet_2d_with_dropout',
                             training_ph, self.init_channels)

        return [x_ph], logits