from .brats import Brats2015, Brats2017
from .isles import *
from .whitematter import WhiteMatterHyperintensity

dataset_name2dataset = {
    'brats2015': Brats2015,
    'brats2017': Brats2017,

    'isles2017': Isles2017,
    'isles2017_augm': Isles2017Augmented,
    'isles2017_crop': Isles2017Crop,
    'isles2017_crop_augm': Isles2017CropAugmented,

    'isles_siss': siss_factory('siss2015.csv'),
    'isles_siss_augmented': siss_factory('augmented_siss.csv'),
    'isles_spes': spes_factory('spes2015.csv'),
    'isles_spes_augmented_core': spes_factory('augmented_spes_core.csv'),
    'isles_spes_augmented_penumbra': spes_factory(
        'augmented_spes_penumbra.csv'),
    'wmhs': WhiteMatterHyperintensity,
}