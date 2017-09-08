from .brats import Brats2017
from .isles import *
from .whitematter import WhiteMatterHyperintensity

from .wrappers import make_cached, make_bbox_extraction, make_normalized, \
    make_normalized_sub, add_groups

name2dataset = {
    'brats2017': Brats2017,

    'isles2017': Isles2017,
    # 'isles2017': Isles2017Raw,
    # 'isles2017_augm': set_filename(Isles2017, 'isles2017_augmented.csv'),
    # 'isles2017_crop': set_filename(Isles2017, 'isles2017_crop.csv'),
    # 'isles2017_crop_augm': set_filename(Isles2017, 'isles2017_crop_augm.csv'),
    # 'isles2017_scaled': Isles2017Scaled,
    # 'isles2017_box': Isles2017Box,
    # 'isles2017_stack': Isles2017Stack,
    #
    # 'isles_siss': set_filename(IslesSISS, 'siss2015.csv'),
    # 'isles_siss_augmented': set_filename(IslesSISS, 'augmented_siss.csv'),
    # 'isles_spes': set_filename(IslesSPES, 'spes2015.csv'),
    # 'isles_spes_augmented_core': set_filename(
    #     IslesSPES, 'augmented_spes_core.csv'),
    # 'isles_spes_augmented_penumbra': set_filename(
    #     IslesSPES, 'augmented_spes_penumbra.csv'),

    'wmhs': WhiteMatterHyperintensity,
}

name2dataset_wrapper = {
    'cached': make_cached,
    'normalized': make_normalized,
    'normalized_sub': make_normalized_sub,
    'bbox_extraction': make_bbox_extraction,
    'groups': add_groups,
}
