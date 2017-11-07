from .patch_3d_stratified import *
from dpipe.config import register


@register()
def patch_3d_stratified(ids, load_x, load_y, *, x_patch_sizes, y_patch_size, nonzero_fraction):
    x_patch_sizes = np.array(x_patch_sizes)
    y_patch_size = np.array(y_patch_size)

    spatial_dims = [-3, -2, -1]

    random_seq = iter(partial(random.choice, ids), None)

    @lru_cache(len(ids))
    def _load_patient(patient_id):
        return Patient(patient_id, load_x(patient_id), load_y(patient_id))

    @lru_cache(len(ids))
    def _find_cancer_and_padding_values(patient: Patient):
        return find_cancer_and_padding_values(patient.x, patient.y, y_patch_size=y_patch_size,
                                              spatial_dims=spatial_dims)

    big_x_patch_size = np.max(x_patch_sizes, axis=0)
    big_x_patch_center_idx = big_x_patch_size // 2

    @pdp.pack_args
    def _extract_big_patches(x, y, cancer_center_indices, padding_values):
        return extract_big_patches(x, y, cancer_center_indices=cancer_center_indices, padding_values=padding_values,
                                   nonzero_fraction=nonzero_fraction, big_x_patch_size=big_x_patch_size,
                                   y_patch_size=y_patch_size, spatial_dims=spatial_dims)

    @pdp.pack_args
    def _extract_patches(x, y):
        return extract_patches(x, y, big_x_patch_center_idx=big_x_patch_center_idx, x_patch_sizes=x_patch_sizes,
                               spatial_dims=spatial_dims)

    return (
        pdp.Source(random_seq, buffer_size=3),
        pdp.One2One(_load_patient, buffer_size=len(ids)),
        pdp.One2One(_find_cancer_and_padding_values, buffer_size=len(ids)),
        pdp.One2One(_extract_big_patches, buffer_size=3),
        pdp.One2One(_extract_patches, buffer_size=3),
    )


@register()
def y_to_volume(x, y):
    return x, [y.sum().astype('float32')]
