import tensorflow as tf


class Scalar:
    def __init__(self, stats: dict, name, writer):
        self.writer = writer

        with tf.variable_scope(name):
            self.summaries = [tf.summary.scalar(k, v) for k, v in stats.items()]

            self.summary_op = tf.summary.merge(self.summaries)

        self.iter = 0

    def write_summary(self, summary):
        self.writer.add_summary(summary, self.iter)
        self.iter += 1


class Custom:
    def __init__(self, name, writer):
        self.writer = writer
        self.name = name

        self.iter = 0

    def write_summary(self, value):
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=self.name,
                                    simple_value=value)])

        self.writer.add_summary(summary, global_step=self.iter)
        self.iter += 1
