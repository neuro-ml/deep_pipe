from .patch_3d_stratified import make_3d_patch_stratified_iter
from .patch_3d_stratified_augm import make_3d_augm_patch_stratified_iter
from .simple import make_simple_iter
from .slices import make_slices_iter, make_multiple_slices_iter


name2batch_iter = {
    '3d_patch_strat': make_3d_patch_stratified_iter,
    '3d_augm_patch_strat': make_3d_augm_patch_stratified_iter,
    'slices': make_slices_iter,
    'simple': make_simple_iter,
    'multiple_slices': make_multiple_slices_iter,
}
