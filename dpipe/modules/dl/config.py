from dpipe.medim.utils import calc_max_dices
from .utils import *

loss_name2loss = {
    'sparse_softmax_cross_entropy': sparse_softmax_cross_entropy,
    'sigmoid_cross_entropy': sigmoid_cross_entropy,
    'soft_dice_loss': soft_dice_loss,
}

predictor_name2predictor = {
    'softmax': softmax,
    'sigmoid': sigmoid
}

metric_name2metric = {
    'max_dices': calc_max_dices
}
