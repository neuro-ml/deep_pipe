from .brats import Brats2015, Brats2017
from .isles import *
from .factories import set_filename
from .whitematter import WhiteMatterHyperintensity

dataset_name2dataset = {
    'brats2015': Brats2015,
    'brats2017': Brats2017,

    'isles2017': Isles2017Raw,
    'isles2017_augm': set_filename(Isles2017, 'isles2017_augmented.csv'),
    'isles2017_crop': set_filename(Isles2017, 'isles2017_crop.csv'),
    'isles2017_crop_augm': set_filename(Isles2017, 'isles2017_crop_augm.csv'),
    'isles2017_scaled': Isles2017Scaled,
    'isles2017_box': Isles2017Box,
    'isles2017_stack': Isles2017Stack,

    'isles_siss': set_filename(IslesSISS, 'siss2015.csv'),
    'isles_siss_augmented': set_filename(IslesSISS, 'augmented_siss.csv'),
    'isles_spes': set_filename(IslesSPES, 'spes2015.csv'),
    'isles_spes_augmented_core': set_filename(
        IslesSPES, 'augmented_spes_core.csv'),
    'isles_spes_augmented_penumbra': set_filename(
        IslesSPES, 'augmented_spes_penumbra.csv'),

    'wmhs': WhiteMatterHyperintensity,
}

_isles_path = {'data_path': '/nmnt/x04-hdd/ISLES/'}
dataset_name2default_params = {
    'brats2015': {
        'data_path': '/nmnt/x02-ssd/brats2015/processed'},
    'brats2017': {
        'data_path': '/nmnt/x02-ssd/brats2017/processed'},

    'isles2017': _isles_path,
    'isles2017_augm': _isles_path,
    'isles2017_crop': _isles_path,
    'isles2017_scaled': _isles_path,
    'isles2017_box': _isles_path,
    'isles2017_stack': _isles_path,
    'isles2017_crop_augm': _isles_path,

    'isles_siss': _isles_path,
    'isles_siss_augmented': _isles_path,
    'isles_spes': _isles_path,
    'isles_spes_augmented_core': _isles_path,
    'isles_spes_augmented_penumbra': _isles_path,

    'wmhs': {'data_path': '/nmnt/x01-ssd/MICCAI_WMHS/'},
}
