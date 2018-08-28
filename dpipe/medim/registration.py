from os.path import join as jp
import tempfile

import numpy as np
import nibabel as nib
from nipype.interfaces.ants import RegistrationSynQuick


def register_images(moving: np.ndarray, fixed: np.ndarray, transform_type: str = 'a', n_threads: int = 1) -> np.ndarray:
    """
    Apply RegistrationSynQuick to the input images.

    Parameters
    ----------
    moving: np.ndarray
    fixed: np.ndarray
    transform_type: str, optional
         |  t:  translation
         |  r:  rigid
         |  a:  rigid + affine (default)
         |  s:  rigid + affine + deformable syn
         |  sr: rigid + deformable syn
         |  b:  rigid + affine + deformable b-spline syn
         |  br: rigid + deformable b-spline syn
    n_threads: int, optional
        the number of threads used to apply the registration
    """
    with tempfile.TemporaryDirectory() as tempdir:
        template_path = jp(tempdir, 'template.nii.gz')
        moving_path = jp(tempdir, 'moving.nii.gz')
        nib.save(nib.Nifti1Image(fixed, np.eye(4)), template_path)
        nib.save(nib.Nifti1Image(moving, np.eye(4)), moving_path)

        reg = RegistrationSynQuick()
        reg.inputs.fixed_image = template_path
        reg.inputs.moving_image = moving_path
        reg.inputs.num_threads = n_threads
        reg.inputs.transform_type = transform_type
        reg.inputs.output_prefix = jp(tempdir, 'transform')
        reg.run()

        return nib.load(jp(tempdir, 'transformWarped.nii.gz')).get_data()
