import tensorflow as tf


class SummaryLogger:
    def __init__(self, name, stats: dict):
        self.name = name
        self.iter = 0

        with tf.variable_scope(self.name):
            summaries = [tf.summary.scalar(k, v) for k, v in stats.items()]
            self.summary_op = tf.summary.merge(summaries)

    def write(self, summary, file_writer):
        file_writer.add_summary(summary, self.iter)
        self.iter += 1


class CustomSummaryWriter:
    def __init__(self, name, writer):
        self.writer = writer
        self.name = name

        self.iter = 0

    def write(self, value):
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=self.name, simple_value=value)])

        self.writer.add_summary(summary, global_step=self.iter)
        self.iter += 1
