from dpipe.medim.utils import calc_max_dices
from .utils import softmax, sigmoid, sparse_softmax_cross_entropy, \
    sigmoid_cross_entropy

loss_name2loss = {
    'sparse_softmax_cross_entropy': sparse_softmax_cross_entropy,
    'sigmoid_cross_entropy': sigmoid_cross_entropy
}

predictor_name2predictor = {
    'softmax': softmax,
    'sigmoid': sigmoid
}

metric_name2metric = {
    'max_dices': calc_max_dices
}
