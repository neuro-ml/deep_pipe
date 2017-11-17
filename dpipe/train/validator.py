from collections import defaultdict

import numpy as np

from dpipe.medim.utils import load_by_ids
from dpipe.config import register


def evaluate(data, single: dict = None, multiple: dict = None):
    metrics_single = defaultdict(list)
    metrics_multiple = {}
    predictions, ys = [], []

    for y, prediction in data:
        if single:
            for name, metric in single.items():
                metrics_single[name].append(metric(y, prediction))

        if multiple:
            predictions.append(prediction)
            ys.append(y)
        else:
            del y, prediction

    if multiple:
        for name, metric in multiple.items():
            value = metric(ys, predictions)
            metrics_multiple[name] = value

    return metrics_single, metrics_multiple


@register(module_type='validator')
def validate(validate, *, load_x, load_y, ids, single: dict = None, multiple: dict = None):
    ys, predictions, losses = [], [], []

    for x, y in load_by_ids(load_x, load_y, ids):
        prediction, loss = validate(x, y)
        ys.append(y)
        predictions.append(prediction)
        losses.append(loss)

    metrics_single, metrics_multiple = evaluate(zip(ys, predictions), single, multiple)
    return losses, metrics_single, metrics_multiple
