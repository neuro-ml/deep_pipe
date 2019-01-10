from typing import Sequence, Callable, Iterable

from dpipe.medim.utils import load_by_ids


def evaluate(y_true: Sequence, y_pred: Sequence, metrics: dict) -> dict:
    """
    Calculates the metrics listed in the ``metrics`` dict.

    Parameters
    ----------
    y_true
    y_pred
    metrics
        a dict, where the key is the metric's name and the value is a
        callable with the standard sklearn signature: (y_true, y_pred) -> metric
    """
    return {name: metric(y_true, y_pred) for name, metric in metrics.items()}


def evaluate_predict(predict: Callable, xs: Iterable, ys_true: Sequence, metrics: dict):
    """
    Evaluate predict function according to metrics

    Parameters
    ----------
    predict: Callable(x) -> prediction
    xs
    ys_true
    metrics
        a dict, where the key is the metric's name and the value is a
        callable with the standard sklearn signature: (y_true, y_pred) -> metric_value

    Returns
    -------
    calculated_metrics: dict
    """
    return evaluate(ys_true, [predict(x) for x in xs], metrics)


def compute_metrics(predict: Callable, load_x: Callable, load_y: Callable, ids: Sequence[str], metrics: dict):
    return evaluate(list(map(load_y, ids)), [predict(load_x(i)) for i in ids], metrics)
