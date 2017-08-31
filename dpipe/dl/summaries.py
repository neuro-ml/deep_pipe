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


def _write_scalar(scalar, step, name, writer):
    s = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=scalar)])
    writer.add_summary(s, global_step=step)


def _write_value(value, step, name, writer):
    if hasattr(value, '__iter__'):
        for i, s in enumerate(value):
            _write_scalar(s, step, name + '/{}'.format(i), writer)
    else:
        _write_scalar(value, step, name, writer)


def make_write_value(name, writer):
    step = 0

    def write_value(value):
        nonlocal step
        _write_value(value, step, name, writer)
        step += 1

    return write_value
