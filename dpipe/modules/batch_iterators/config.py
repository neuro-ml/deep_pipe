from .patch_3d_stratified import make_3d_patch_stratified_iter
from .slices import make_slices_iter

batch_iter_name2batch_iter = {
    '3d_patch_strat': make_3d_patch_stratified_iter,
    'slices': make_slices_iter,
}
