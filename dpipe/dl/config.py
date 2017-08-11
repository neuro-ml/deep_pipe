from dpipe.medim.utils import calc_max_dices
from .utils import softmax, sigmoid, sparse_softmax_cross_entropy, \
    sigmoid_cross_entropy

name2loss = {
    'sparse_softmax_cross_entropy': sparse_softmax_cross_entropy,
    'sigmoid_cross_entropy': sigmoid_cross_entropy
}

name2predictor = {
    'softmax': softmax,
    'sigmoid': sigmoid
}

name2metric = {
    'max_dices': calc_max_dices
}
