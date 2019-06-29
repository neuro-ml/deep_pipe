from typing import Sequence, Callable, Iterable


def evaluate(y_true: Sequence, y_pred: Sequence, metrics: dict) -> dict:
    """
    Calculates the metrics listed in the ``metrics`` dict.

    Parameters
    ----------
    y_true: Sequence
        sequence of ground truth objects.
    y_pred: Sequence
        sequence of predicted object.
    metrics: dict
        ``dict`` metric names as keys and
        ``callable`` as values with the standard sklearn signature: (y_true, y_pred) -> metric

    Returns
    -------
    calculated_metrics: dict
    """
    return {name: metric(y_true, y_pred) for name, metric in metrics.items()}


def evaluate_predict(predict: Callable, xs: Iterable, ys_true: Sequence, metrics: dict):
    """
    Evaluate predict function according to metrics

    Parameters
    ----------
    predict: Callable(x) -> prediction
        function to return prediction from the input element.
    xs: Iterable
        iterator to return input elements for ``predict``.
    ys_true: Sequence
        sequence of ground truth objects, corresponding to ``xs`` input elements.
        Should be the same size with ``xs``.
    metrics: dict
        ``dict`` metric names as keys and
        ``callable`` as values with the standard sklearn signature: (y_true, y_pred) -> metric

    Returns
    -------
    calculated_metrics: dict
    """
    return evaluate(ys_true, [predict(x) for x in xs], metrics)


def compute_metrics(predict: Callable, load_x: Callable, load_y: Callable, ids: Sequence[str], metrics: dict):
    return evaluate(list(map(load_y, ids)), [predict(load_x(i)) for i in ids], metrics)
