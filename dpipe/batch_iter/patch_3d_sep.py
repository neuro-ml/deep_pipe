from .patch_3d import *
from dpipe.config import register


@register()
def patch_3d_part(ids, load_x, load_y, *, x_patch_sizes, y_patch_size):
    x_patch_sizes = np.array(x_patch_sizes)
    y_patch_size = np.array(y_patch_size)

    spatial_dims = [-3, -2, -1]

    random_seq = iter(functools.partial(random.choice, ids), None)

    @functools.lru_cache(len(ids))
    def _load_patient(patient_id):
        return Patient(patient_id, load_x(patient_id), load_y(patient_id))

    @functools.lru_cache(len(ids))
    def _find_padding_values(patient: Patient):
        return patient.x, patient.y, np.min(patient.x, axis=tuple(spatial_dims), keepdims=True)

    @pdp.pack_args
    def _extract_patches(x, y, padding_values):
        center_idx = get_random_center_idx(y, y_patch_size, spatial_dims=spatial_dims)

        xs = extract_patches(x, patch_sizes=x_patch_sizes, center_idx=center_idx, padding_values=padding_values,
                             spatial_dims=spatial_dims)
        y, = extract_patches(y, patch_sizes=[y_patch_size], center_idx=center_idx, padding_values=0,
                             spatial_dims=spatial_dims)

        return (*xs, y)

    return (
        pdp.Source(random_seq, buffer_size=3),
        pdp.One2One(_load_patient, buffer_size=len(ids)),
        pdp.One2One(_find_padding_values, buffer_size=len(ids)),
        pdp.One2One(_extract_patches, buffer_size=3),
    )


@register()
def y_to_volume(x, y):
    return x, [y.sum().astype('float32')]
