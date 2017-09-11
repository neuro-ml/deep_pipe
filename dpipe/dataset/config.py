from .brats import Brats2017
from .isles import *
from .whitematter import WhiteMatterHyperintensity, Wmh2017

from .wrappers import make_cached, make_bbox_extraction, make_normalized, \
    make_normalized_sub, add_groups

name2dataset = {
    'brats2017': Brats2017,

    'isles2017': Isles2017,

    'wmhs': WhiteMatterHyperintensity,
    'wmh2017': Wmh2017
}

name2dataset_wrapper = {
    'cached': make_cached,
    'normalized': make_normalized,
    'normalized_sub': make_normalized_sub,
    'bbox_extraction': make_bbox_extraction,
    'groups': add_groups,
}
