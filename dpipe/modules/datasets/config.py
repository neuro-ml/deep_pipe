from dpipe.modules.datasets.factories import FromDataFrame
from .brats import Brats2015, Brats2017
from .isles import *
from .whitematter import WhiteMatterHyperintensity

dataset_name2dataset = {
    'brats2015': Brats2015,
    'brats2017': Brats2017,

    'isles2017': Isles2017,
    'isles2017_augm': ISles2017Augmented,
    'isles2017_crop': Isles2017Crop,
    'isles2017_crop_augm': ISles2017CropAugmented,

    'isles_siss': siss_factory('siss2015.csv'),
    'isles_siss_augmented': siss_factory('augmented_siss.csv'),
    'isles_spes': spes_factory('spes2015.csv'),
    'isles_spes_augmented_core': spes_factory('augmented_spes_core.csv'),
    'isles_spes_augmented_penumbra': spes_factory(
        'augmented_spes_penumbra.csv'),
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
    'isles2017_crop_augm': _isles_path,

    'isles_siss': {'data_path': '/nmnt/x04-hdd/ISLES/'},
    'isles_siss_augmented': {'data_path': '/nmnt/x04-hdd/ISLES/'},
    'isles_spes': {'data_path': '/nmnt/x04-hdd/ISLES/'},
    'isles_spes_augmented_core': {'data_path': '/nmnt/x04-hdd/ISLES/'},
    'isles_spes_augmented_penumbra': {'data_path': '/nmnt/x04-hdd/ISLES/'},

    'wmhs': {'data_path': '/nmnt/x01-ssd/MICCAI_WMHS/'},
}
