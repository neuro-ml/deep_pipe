import numpy as np

from dpipe.batch_predict.base import DivideCombine


def add_dimension(*data):
    result = tuple(x[None] for x in data)
    if len(result) == 1:
        result = result[0]
    yield result


def extract_dimension(predictions):
    predictions = list(predictions)
    assert len(predictions) == 1 and len(predictions[0]) == 1
    return predictions[0][0]


class Simple(DivideCombine):
    def __init__(self):
        super().__init__(add_dimension, extract_dimension)


class MultiClass(DivideCombine):
    def __init__(self):
        super().__init__(add_dimension, lambda x: np.argmax(extract_dimension(x), axis=0))
