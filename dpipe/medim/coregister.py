import shlex
import subprocess
import os
import shutil
from typing import Sequence

invert = 'ImageMath 3 %s Neg %s'
register = 'antsRegistrationSyNQuick.sh -d 3 -m %s -f %s -t s -o result -x %s'
transform = 'antsApplyTransforms -d 3 -i %s -o %s -r ' + \
            'resultWarped.nii.gz -t result1InverseWarp.nii.gz'


# TODO: split into several smaller modules
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
        Path to the reference image which is used as the coregistration taget

    Notes
    -----
    Be sure to add ANTSPATH to the path before running the script
    """
    os.makedirs(result_path, exist_ok=True)

    masks_paths = list(map(os.path.abspath, masks_paths))
    modalities_paths = list(map(os.path.abspath, modalities_paths))
    result_path = os.path.abspath(result_path)

    neg_masks = [os.path.join(result_path, 'neg_mask%d.nii.gz' % i)
                 for i in range(len(masks_paths))]
    for neg_mask, mask in zip(neg_masks, masks_paths):
        command = invert % (neg_mask, mask)
        subprocess.call(shlex.split(command), cwd=result_path)

    mask_files = list(map(os.path.basename, masks_paths))
    mod_files = list(map(os.path.basename, modalities_paths))
    if not mod_files[0].endswith('gz'):
        mod_files[0] = mod_files[0] + '.gz'

    # apparently ANTs can't handle comas, so create a symlink:
    filename = ''
    if ',' in ref_path:
        filename = os.path.basename(ref_path).replace(',', '')
        filename = os.path.join(result_path, filename)
        os.symlink(ref_path, filename)
        ref_path = filename

    # create the transformation
    command = register % (ref_path, modalities_paths[0], neg_masks[0])
    subprocess.call(shlex.split(command), cwd=result_path)
    warped = os.path.join(result_path, 'resultWarped.nii.gz')
    shutil.copyfile(warped, os.path.join(result_path, mod_files[0]))

    # TODO: the first coregistered brain is somewhat different from others
    # create other modalities
    for file, name in zip(modalities_paths[1:], mod_files[1:]):
        command = transform % (file, name)
        subprocess.call(shlex.split(command), cwd=result_path)

    # create the masks
    for neg_mask, name in zip(neg_masks, mask_files):
        output = '_' + name
        command = transform % (neg_mask, output)
        subprocess.call(shlex.split(command), cwd=result_path)
        command = invert % (name, output)
        subprocess.call(shlex.split(command), cwd=result_path)
        os.remove(os.path.join(result_path, output))

    if ref_path == filename:
        os.unlink(ref_path)

    for neg_mask in neg_masks:
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
