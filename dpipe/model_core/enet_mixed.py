import numpy as np
import tensorflow as tf

from .base import ModelCore
from .layers import spatial_batch_norm
from dpipe import medim


def init_block(inputs, name, training, inp_channels, output_channels,
               kernel_size=3,
               conv_layer=tf.layers.conv2d, pool_layer=tf.layers.max_pooling2d):
    with tf.variable_scope(name):
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
              pool_layer=tf.layers.max_pooling2d, strides=2):
    # it can be either upsampling or downsampling:
    assert not (upsample and downsample)

    input_channels = int(inputs.shape[1])
    internal_channels = output_channels // internal_scale
    input_stride = strides if downsample else 1

    with tf.variable_scope(name):
        # conv path
        # TODO: use prelu
        conv = conv_block(inputs, internal_channels, kernel_size=input_stride,
                          strides=input_stride, training=training, name='conv1',
                          layer=conv_down)

        if upsample:
            conv = conv_block(conv, internal_channels, kernel_size,
                              strides=strides, training=training,
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
                inputs, pool_size=2, strides=strides,
                padding='same', data_format='channels_first')
        if output_channels != input_channels:
            main = conv_block(main, output_channels, kernel_size=1, strides=1,
                              activation=tf.identity, training=training,
                              name='justify', layer=conv_down)
        if upsample:
            main = conv_up(
                main, output_channels, kernel_size, strides=strides,
                padding='same',
                use_bias=False, data_format='channels_first',
                name='justify_upsample')
        return tf.nn.relu(conv + main)


def stage(inputs, output_channels, num_blocks, name, training,
          downsample=False, upsample=False, conv_down=tf.layers.conv2d,
          conv_up=tf.layers.conv2d_transpose,
          pool_layer=tf.layers.max_pooling2d, strides=2):
    layers = dict(conv_down=conv_down, conv_up=conv_up, pool_layer=pool_layer,
                  strides=strides)
    with tf.variable_scope(name):
        inputs = res_block(inputs, 'res0', output_channels, training=training,
                           downsample=downsample, upsample=upsample, **layers)
        for i in range(num_blocks - 1):
            inputs = res_block(inputs, f'res%d' % (i + 1), output_channels,
                               training=training, **layers)
        return inputs


def build_model(inputs, classes, name, training, init_channels):
    layers = dict(conv_down=tf.layers.conv3d,
                  conv_up=tf.layers.conv3d_transpose,
                  pool_layer=tf.layers.max_pooling3d)
    with tf.variable_scope(name):
        input_shape = tf.shape(inputs)[2:]
        rest = tf.shape(inputs)[1:]
        input_channels = int(inputs.shape[1])

        inputs = tf.transpose(inputs, perm=[0, 4, 1, 2, 3])
        inputs = tf.reshape(inputs, [-1, input_channels, rest[1], rest[2]])

        inputs = init_block(inputs, 'init', training, input_channels,
                            init_channels)
        inputs = stage(inputs, 64, 5, 'stage1', training, downsample=True)
        inputs = stage(inputs, 128, 9, 'stage2', training, downsample=True)

        current = tf.shape(inputs)[1:]
        channels = int(inputs.shape[1])
        inputs = tf.reshape(inputs,
                            [-1, rest[3], channels, current[1], current[2]])

        inputs = tf.transpose(inputs, perm=[0, 2, 3, 4, 1])

        inputs = stage(inputs, 128, 8, 'stage3', training, **layers,
                       downsample=True, strides=(1, 1, 2))
        inputs = stage(inputs, 128, 3, 'stage4', training, upsample=True,
                       **layers, strides=(2, 2, 1))
        inputs = stage(inputs, 64, 2, 'stage5', training, upsample=True,
                       **layers)
        inputs = stage(inputs, 16, 3, 'stage6', training, upsample=True,
                       **layers, strides=(2, 2, 1))

        inputs = tf.layers.conv3d_transpose(
            inputs, classes, kernel_size=2, strides=(2, 2, 1), use_bias=False,
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


def build_model2(inputs, classes, name, training, init_channels):
    layers = dict(conv_down=tf.layers.conv3d,
                  conv_up=tf.layers.conv3d_transpose,
                  pool_layer=tf.layers.max_pooling3d)
    with tf.variable_scope(name):
        input_shape = tf.shape(inputs)[2:]
        rest = tf.shape(inputs)[1:]
        input_channels = int(inputs.shape[1])

        inputs = tf.transpose(inputs, perm=[0, 4, 1, 2, 3])
        inputs = tf.reshape(inputs, [-1, input_channels, rest[1], rest[2]])

        inputs = init_block(inputs, 'init', training, input_channels,
                            init_channels)
        inputs = stage(inputs, 64, 5, 'stage1', training, downsample=True)
        # inputs = stage(inputs, 128, 9, 'stage2', training, downsample=True)

        current = tf.shape(inputs)[1:]
        channels = int(inputs.shape[1])
        inputs = tf.reshape(inputs,
                            [-1, rest[3], channels, current[1], current[2]])

        inputs = tf.transpose(inputs, perm=[0, 2, 3, 4, 1])
        inputs = stage(inputs, 128, 8, 'stage3', training, **layers,
                       downsample=True, strides=(1, 1, 2))
        # inputs = stage(inputs, 128, 3, 'stage4', training, upsample=True,
        #                **layers, strides=(2,2,1))
        inputs = stage(inputs, 64, 2, 'stage5', training, upsample=True,
                       **layers)
        inputs = stage(inputs, 16, 3, 'stage6', training, upsample=True,
                       **layers, strides=(2, 2, 1))
        inputs = tf.layers.conv3d_transpose(
            inputs, classes, kernel_size=2, strides=(2, 2, 1), use_bias=False,
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


class ENetMixed(ModelCore):
    def build(self, training_ph):
        x_ph = tf.placeholder(
            tf.float32, (None, self.n_chans_in, None, None, None), name='input'
        )

        logits = build_model(x_ph, self.n_chans_out, 'enet_mixed',
                             training_ph, 16)
        return [x_ph], logits

    def validate_object(self, x, y, do_val_step):
        return do_val_step(x[None], y[None])

    def predict_object(self, x, do_inf_step):
        return do_inf_step(x[None])


class ENetPatch(ModelCore):
    def __init__(self, *, n_chans_in, n_chans_out, n_parts):
        super().__init__(n_chans_in=n_chans_in, n_chans_out=n_chans_out)

        self.kernel_size = 3
        self.n_parts = np.array(n_parts)
        assert np.all((self.n_parts == 1) | (self.n_parts == 2))

    def build(self, training_ph):
        x_ph = tf.placeholder(
            tf.float32, (None, self.n_chans_in, None, None, None), name='input'
        )

        logits = build_model2(x_ph, self.n_chans_out, 'enet_mixed',
                              training_ph, 16)
        return [x_ph], logits

    def validate_object(self, x, y, do_val_step):
        x_parts = medim.divide.divide_no_padding(x, [0] * 4,
                                                 n_parts_per_axis=[1, *self.n_parts])
        y_parts = medim.divide.divide_no_padding(y, [0] * 4,
                                                 n_parts_per_axis=[1, *self.n_parts])

        y_pred_parts, weights, losses = [], [], []
        for x_part, y_part in zip(x_parts, y_parts):
            y_pred, loss = do_val_step(x_part[None, :], y_part[None, :])
            y_pred_parts.append(y_pred[0])
            losses.append(loss)
            weights.append(y_pred.size)

        loss = np.average(losses, weights=weights)
        y_pred = medim.divide.combine(y_pred_parts, [1, *self.n_parts])
        return y_pred, loss

    def predict_object(self, x, do_inf_step):
        x_parts = medim.divide.divide_no_padding(x, [0] * 4,
                                                 n_parts_per_axis=[1, *self.n_parts])

        y_pred_parts = []
        for x_part in x_parts:
            y_pred = do_inf_step(x_part[None, :])[0]
            y_pred_parts.append(y_pred)

        y_pred = medim.divide.combine(y_pred_parts, [1, *self.n_parts])
        return y_pred
