from ..medim.utils import calc_max_dices
from .utils import softmax, sigmoid, optimize, sparse_softmax_cross_entropy, \
    sigmoid_cross_entropy, soft_dice_loss
from .model_controller import ModelController
from .model import Model, FrozenModel

name2loss = {
    'sparse_softmax_cross_entropy': sparse_softmax_cross_entropy,
    'sigmoid_cross_entropy': sigmoid_cross_entropy,
    'soft_dice_loss': soft_dice_loss,
}

name2predict = {
    'softmax': softmax,
    'sigmoid': sigmoid
}

name2metric = {
    'max_dices': calc_max_dices
}

name2optimize = {
    'tf_optimize': optimize
}

name2model = {
    'model': Model,
    'frozen_model': FrozenModel
}

name2model_controller = {
    'model_controller': ModelController
}
