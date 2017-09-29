import numpy as np
import tensorflow as tf


def infer_spatial_size(x: tf.Tensor):
    return len(x.get_shape()) - 2


def check_data_format(data_format):
    if data_format != 'channels_first' and data_format != 'channels_last':
        raise ValueError(
            f'Unknown data format: {data_format}\n'
            'Possible values: "channels_first" and "channels_last"'
        )


def expand_size(size, ndim):
    if not hasattr(size, '__len__'):
        size = [size] * ndim
    else:
        assert len(size) == ndim

    return size
