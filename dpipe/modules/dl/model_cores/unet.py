import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from dpipe.modules.dl import ModelCore
from dpipe.modules.batch_iterators.slices import \
    iterate_multiple_slices as iterate_slices


def conv_block(inputs, output_channels, training, name, kernel_size=3,
               strides=1, layer=tf.layers.conv2d):
    with tf.variable_scope(name):
        for i in range(2):
            inputs = layer(inputs, output_channels, kernel_size=kernel_size,
                           strides=strides, padding='same', use_bias=False,
                           data_format='channels_first', name=f'conv_{i}')
            inputs = slim.batch_norm(inputs, decay=0.9, scale=True,
                                     is_training=training,
                                     data_format='NCHW', fused=True)
            inputs = tf.nn.relu(inputs)
    return inputs


def upsample_concat(lower, upper, name):
    with tf.variable_scope(name):
        # channels = int(lower.shape[1])
        # lower = tf.layers.conv2d_transpose(
        #     lower, channels, kernel_size=kernel_size, strides=2,
        # use_bias=False,
        #     data_format='channels_first', name='conv')
        lower = tf.transpose(lower, perm=[0, 2, 3, 1])
        lower = tf.image.resize_images(lower, tf.shape(upper)[-2:])
        lower = tf.transpose(lower, perm=[0, 3, 1, 2])

        return tf.concat([lower, upper], axis=1, name='concat')


def build_model(inputs, classes, name, training):
    channels = [16, 32]
    bridge = channels[-1]

    with tf.variable_scope(name):
        # down path
        down = []
        for i, channel in enumerate(channels):
            inputs = conv_block(inputs, channel, training, f'down_{i}')
            down.append(inputs)

            inputs = tf.layers.max_pooling2d(
                inputs, pool_size=2, strides=2, padding='same',
                data_format='channels_first', name=f'pool_{i}')

        # bridge:
        inputs = conv_block(inputs, bridge, training, f'bridge')

        # up path
        for i, (ch, lower) in enumerate(
                zip(reversed(channels), reversed(down))):
            inputs = upsample_concat(inputs, lower, f'upsample_{i}')
            inputs = conv_block(inputs, ch, training, f'up_{i}',
                                layer=tf.layers.conv2d_transpose)

        inputs = tf.layers.conv2d(inputs, classes, kernel_size=1,
                                  use_bias=True, data_format='channels_first')
        return inputs


class UNet2D(ModelCore):
    def __init__(self, n_chans_in, n_chans_out, multiplier=1, init_channels=16):
        super().__init__(n_chans_in * multiplier, n_chans_out)
        self.init_channels = init_channels
        self.multiplier = multiplier

    def build(self, training_ph):
        x_ph = tf.placeholder(
            tf.float32, (None, self.n_chans_in, None, None), name='input'
        )

        logits = build_model(x_ph, self.n_chans_out, 'unet_2d', training_ph)
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
