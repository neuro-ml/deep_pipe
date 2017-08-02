from dpipe.datasets import FromDataFrame, Scaled, Padded


# TODO: create filename factory

class IslesSPES(Scaled):
    modality_cols = ['CBF', 'CBV', 'DWI', 'T1c', 'T2', 'TTP', 'Tmax']
    target_cols = ['penumbralabel', 'corelabel']
    global_path = False
    spacial_shape = 96, 110, 72
    axes = -3, -2, -1
    group_col = 'patient'


class IslesSISS(Padded):
    modality_cols = ['T1', 'T2', 'Flair', 'DWI']
    target_cols = ['OT']
    global_path = False
    spacial_shape = 230, 230, 154
    axes = -3, -2, -1
    group_col = 'patient'


def spes_factory(_filename):
    class Child(IslesSPES):
        filename = _filename

    return Child


def siss_factory(_filename):
    class Child(IslesSISS):
        filename = _filename

    return Child


class Isles2017(Scaled):
    modality_cols = ['ADC', 'MTT', 'TTP', 'Tmax', 'rCBF', 'rCBV']
    target_cols = ['OT']
    filename = 'meta2017.csv'
    global_path = False
    spacial_shape = 192, 192
    axes = -3, -2


class Isles2017Augmented(Isles2017):
    filename = 'isles2017_augmented.csv'
    group_col = 'patient'


class Isles2017Crop(FromDataFrame):
    modality_cols = ['ADC', 'MTT', 'TTP', 'Tmax', 'rCBF', 'rCBV']
    target_cols = ['OT']
    filename = 'isles2017_crop.csv'
    global_path = False


class Isles2017CropAugmented(Isles2017Crop):
    filename = 'isles2017_crop_augm.csv'
    group_col = 'patient'
