# TODO: this is a script not a submodule. I guess it should be deprecated


import os
import shlex
import subprocess
from typing import Sequence

import nibabel as nib


def invert_mask(input_path, output_path):
    command = 'ImageMath 3 %s Neg %s'
    command = command % (output_path, input_path)
    subprocess.check_call(shlex.split(command))


def create_transformation(source, reference, neg_mask, result_folder):
    command = 'antsRegistrationSyNQuick.sh -d 3 -m %s -f %s -t s -o result -x %s'
    command = command % (reference, source, neg_mask)
    subprocess.check_call(shlex.split(command), cwd=result_folder)


def apply_transformation(input_path, output_path, transform_folder):
    command = ('antsApplyTransforms -d 3 -i %s -o %s -r '
               'resultWarped.nii.gz -t result1InverseWarp.nii.gz')
    command = command % (input_path, output_path)
    subprocess.check_call(shlex.split(command), cwd=transform_folder)


def copy_header(image, data):
    """
    Creates a new nifty image from `data`, using the `image`'s header.

    Parameters
    ----------
    image: nibabel Nifti image
    data: np.array
    """
    header = image.header
    if header['sizeof_hdr'] == 348:
        constructor = nib.Nifti1Image
    elif header['sizeof_hdr'] == 540:
        constructor = nib.Nifti2Image
    else:
        raise TypeError('Unrecognized image header')

    result = constructor(data, image.affine, header=header)
    result.set_data_dtype(data.dtype)
    return result


def coregister(result_path: str, masks_paths: Sequence[str], modalities_paths: Sequence[str], ref_path: str):
    """
    Calculates a coregistration transformation and applies it to all the images
    taking the lesion masks into account.

    The initial transformation will be calculated based on the first entry in
    masks_paths and modalities_paths.

    Parameters
    ----------
    result_path: str
        Path to the transformation results folder
    masks_paths: Sequence[str]
        Paths to the lesion masks. Note that the first mask will be used to calculate
        the coregistration mapping.
    modalities_paths: Sequence[str]
        Paths to the brain images. Note that the first image will be used to calculate
        the coregistration mapping.
    ref_path: str
        Path to the reference image which is used as the coregistration target

    Notes
    -----
    Be sure to add ANTSPATH to the path before running the script
    """
    os.makedirs(result_path, exist_ok=True)

    masks_paths = list(map(os.path.abspath, masks_paths))
    modalities_paths = list(map(os.path.abspath, modalities_paths))
    result_path = os.path.abspath(result_path)
    neg_masks = [os.path.join(result_path, 'neg_mask%d.nii.gz' % i) for i in range(len(masks_paths))]

    for neg_mask, mask in zip(neg_masks, masks_paths):
        invert_mask(mask, neg_mask)

    create_transformation(modalities_paths[0], ref_path, neg_masks[0], result_path)

    for file in modalities_paths:
        name = os.path.basename(file)
        apply_transformation(file, name, result_path)

    for neg_mask, orig_mask in zip(neg_masks, masks_paths):
        name = os.path.basename(neg_mask)
        output = '_' + name
        apply_transformation(neg_mask, output, result_path)

        # invert the masks and threshold the result
        template = nib.load(os.path.join(result_path, output))
        data = template.get_data()
        mask = copy_header(template, (data <= .5).astype('uint8'))
        nib.save(mask, os.path.join(result_path, os.path.basename(orig_mask)))

        os.remove(os.path.join(result_path, output))
        os.remove(neg_mask)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('result_path')
    parser.add_argument('-msk', '--masks_paths', nargs='+')
    parser.add_argument('-mod', '--modalities_paths', nargs='+')
    parser.add_argument('-ref', '--ref_path')
    args = parser.parse_args()

    coregister(**dict(args._get_kwargs()))
