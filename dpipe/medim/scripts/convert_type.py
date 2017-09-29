import argparse

import numpy as np
import nibabel as nib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type')
    parser.add_argument('filepath')

    args = parser.parse_args()
    filepath = args.filepath
    dtype = getattr(np, args.type)

    image = nib.load(filepath)
    new_data = np.array(image.get_data(), dtype=dtype)
    hd = image.header

    # if nifty1
    if hd['sizeof_hdr'] == 348:
        new_image = nib.Nifti1Image(new_data, image.affine, header=hd)
    # if nifty2
    elif hd['sizeof_hdr'] == 540:
        new_image = nib.Nifti2Image(new_data, image.affine, header=hd)
    else:
        raise IOError('Input image header problem')
    new_image.set_data_dtype(dtype)
    nib.save(new_image, filepath)