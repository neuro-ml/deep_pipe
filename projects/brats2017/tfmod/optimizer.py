import tensorflow as tf
import tensorflow.contrib.slim as slim


class Optimizer:
    def __init__(self, loss):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        with tf.variable_scope('optimization'):
            opt = tf.train.MomentumOptimizer(self.lr, 0.9, use_nesterov=True)
            self.train_op = slim.learning.create_train_op(loss, opt)
