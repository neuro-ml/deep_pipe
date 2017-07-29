import numpy as np

from dpipe.medim.bounding_box import get_slice
from dpipe.modules.datasets.factories import *
from dpipe.medim.preprocessing import scale, pad


# 2015

@msegm(scale, spacial_shape=(96, 110, 72), order=0)
@mscan(scale, spacial_shape=(96, 110, 72), order=3)
class IslesSPES(FromDataFrame):
    modality_cols = ['CBF', 'CBV', 'DWI', 'T1c', 'T2', 'TTP', 'Tmax']
    target_cols = ['penumbralabel', 'corelabel']
    group_col = 'patient'


@apply(pad, spacial_shape=(230, 230, 154))
class IslesSISS(FromDataFrame):
    modality_cols = ['T1', 'T2', 'Flair', 'DWI']
    target_cols = ['OT']
    group_col = 'patient'


# 2017

class Isles2017(FromDataFrame):
    modality_cols = ['ADC', 'MTT', 'TTP', 'Tmax', 'rCBF', 'rCBV']
    target_cols = ['OT']
    group_col = 'patient'


@msegm(scale, spacial_shape=(192, 192), axes=(-3, -2), order=0)
@mscan(scale, spacial_shape=(192, 192), axes=(-3, -2), order=3)
class Isles2017Raw(Isles2017):
    filename = 'meta2017.csv'


@apply(pad, spacial_shape=(200, 200, 25))
class Isles2017Crop3D(Isles2017):
    filename = 'isles2017_crop.csv'


def box(y):
    result = []
    for i in range(y.shape[-1]):
        y_ = y[..., i].copy()
        if y_.any():
            y_[get_slice(y_)] = 1
        result.append(y_)
    result = np.stack(result, -1)
    return result


@msegm(box)
class Isles2017Box(Isles2017):
    filename = 'isles2017_crop.csv'


@append_channels
class Isles2017Stack(Isles2017):
    filename = 'isles2017_crop.csv'
