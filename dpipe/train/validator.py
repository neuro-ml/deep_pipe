from typing import Sequence, Callable

from dpipe.medim.utils import load_by_ids


def evaluate(y_true: Sequence, y_pred: Sequence, metrics: dict):
    """
    Calculates the metrics listed in the `metrics` dict.

    Parameters
    ----------
    y_true: ground truth
    y_pred: predictions
    metrics: dict
        a dict, where the key is the metric's name and the value is a
        callable with the standard sklearn signature: (y_true, y_pred) -> metric

    Returns
    -------
    calculated_metrics: dict
    """
    return {name: metric(y_true, y_pred) for name, metric in metrics.items()}


def validate(validate_fn: Callable, load_x: Callable, load_y: Callable, ids: Sequence[str], metrics: dict = None):
    """
    Performs a validation step.

    Parameters
    ----------
    validate_fn: callable(x_batch, y_batch) -> (prediction, loss)
    load_x: callable(id)
    load_y: callable(id)
    ids: Sequence[str]
    metrics: dict
        a dict, where the key is the metric's name and the value is a
        callable with the standard sklearn signature: (y_true, y_pred) -> metric

    Returns
    -------
    losses: [float]
        a list of losses for each pair (x, y)
    calculated_metrics: dict
    """
    ys, predictions, losses = [], [], []

    for x, y in load_by_ids(load_x, load_y, ids=ids):
        prediction, loss = validate_fn(x, y)
        losses.append(loss)
        if metrics:
            ys.append(y)
            predictions.append(prediction)

    return losses, evaluate(ys, predictions, metrics or {})


def compute_metrics(predict: Callable, load_x: Callable, load_y: Callable, ids: Sequence[str], metrics: dict):
    return evaluate(list(map(load_y, ids)), [predict(load_x(i)) for i in ids], metrics)
