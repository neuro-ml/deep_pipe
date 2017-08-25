from .train_with_lr_decrease import train_with_lr_decrease
from .train_segm import train_segm

name2train = {
    'train_with_lr_decrease': train_with_lr_decrease,
    'train_segm': train_segm
}
