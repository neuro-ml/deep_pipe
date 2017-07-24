import shlex
import subprocess
import os
import shutil

invert = 'ImageMath 3 %s Neg %s'
register = 'antsRegistrationSyNQuick.sh -d 3 -m %s -f %s -t s -o result -x %s'
transform = 'antsApplyTransforms -d 3 -i %s -o %s -r ' + \
            'resultWarped.nii.gz -t result1InverseWarp.nii.gz'


def coregister(result_path, masks_paths, modalities_paths, refs_paths):
    """
    The initial transformation will be calculated based on the first entry in
    masks_paths and modalities_paths.

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

    filename = ''
    sym_counter = 0

    for id, image in refs_paths:
        folder = os.path.join(result_path, str(id))
        try:
            os.makedirs(folder)
        except FileExistsError:
            # this patient is already processed, so do nothing
            continue

        # apparently ANTs can't handle comas, so create a symlink:
        if ',' in image:
            sym_counter += 1
            filename = os.path.basename(image).replace(',', '')
            filename = os.path.join(folder, str(sym_counter) + filename)
            os.symlink(image, filename)
            image = filename

        # create the transformation
        command = register % (image, modalities_paths[0], neg_masks[0])
        subprocess.call(shlex.split(command), cwd=folder)
        warped = os.path.join(folder, 'resultWarped.nii.gz')
        shutil.copyfile(warped, os.path.join(folder, mod_files[0]))

        # create other modalities
        for file, name in zip(modalities_paths[1:], mod_files[1:]):
            command = transform % (file, name)
            subprocess.call(shlex.split(command), cwd=folder)

        # create the masks
        for neg_mask, name in zip(neg_masks, mask_files):
            output = '_' + name
            command = transform % (neg_mask, output)
            subprocess.call(shlex.split(command), cwd=folder)
            command = invert % (name, output)
            subprocess.call(shlex.split(command), cwd=folder)
            os.remove(os.path.join(folder, output))

        if image == filename:
            os.unlink(image)

    for neg_mask in neg_masks:
        os.remove(neg_mask)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('results_path')
    parser.add_argument('-msk', '--masks_paths', nargs='+')
    parser.add_argument('-mod', '--modalities_paths', nargs='+')
    parser.add_argument('-ref', '--refs_paths', nargs='+')
    args = parser.parse_args()

    coregister(**dict(args._get_kwargs()))
