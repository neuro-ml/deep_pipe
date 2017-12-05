from dpipe.medim.utils import load_by_ids


def evaluate(y_true, y_pred, metrics: dict):
    return {name: metric(y_true, y_pred) for name, metric in metrics.items()}


def validate(validate_fn, load_x, load_y, ids, metrics=None):
    ys, predictions, losses = [], [], []

    for x, y in load_by_ids(load_x, load_y, ids):
        prediction, loss = validate_fn(x, y)
        ys.append(y)
        predictions.append(prediction)
        losses.append(loss)

    if metrics is None:
        result = {}
    else:
        result = evaluate(ys, predictions, metrics)
    return losses, result
