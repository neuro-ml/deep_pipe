from .cv_111 import get_cv_111, get_group_cv_111, get_pure_val_test_group_cv_111
from .cv_11 import get_cv_11
from .wmhs_los_cv import get_los_cv

name2split = {
    'cv_11': get_cv_11,
    'cv_111': get_cv_111,
    'group_cv_111': get_group_cv_111,
    'group_pure_val_test_cv_111': get_pure_val_test_group_cv_111,
    'wmh_leave_one_tomograph_cv' : get_los_cv,
}
