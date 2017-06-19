"""
Data loading (with padding).
Random crops.
Data augmentation.
Data splitter.
Data combiner.
"""

import os
from itertools import product

import nibabel as nib
from tqdm import tqdm
import numpy as np
import scipy
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import zoom, map_coordinates
from scipy.ndimage.filters import gaussian_filter



def _reshape_to(tmp: np.ndarray, new_shape = None):
    """
    Reshape ND array to new shapes.
    
    Parameters
    ----------
    tmp : np.array
        ND array with shapes less than new_shape.
    new_shape : tuple 
        Tuple from N number - new shape.
    Returns
    ------
    result : np.array
        Return np.array with shapes equal to new_shape.

     Example.
    _ = _reshape_to(X_test[..., :80], (15, 2, 288, 288, 100))
    """
    assert not new_shape is None
    new_diff = [((new_shape[-i] - tmp.shape[-i]) // 2,
                 (new_shape[-i] - tmp.shape[-i]) // 2 + (new_shape[-i] - tmp.shape[-i]) % 2)
                for i in range(len(new_shape), 0, -1)]
    return np.pad(tmp, new_diff, mode='constant', constant_values=0)


def load_files(PATH: str):
    """
    Using structure from WMHS challenge load FLAIR and T1 scans.
    """
    masks = []
    t1 = []
    flairs = []
    clear_flairs = []
    #t = None
    # read data
    path_to_flairs = '/home/mount/neuro-t01-ssd/home/amir/projects/segm/projects/WMH/data/skull_stripping'
    for i in tqdm(os.listdir(PATH)):
        tmp_path_to_flairs = os.path.join(path_to_flairs, PATH.split('/')[-2], i, 'brain_mask/brainmask.nii.gz')
        path_for_wmhm = os.path.join(PATH, i, 'wmh.nii.gz')
        masks.append(nib.load(path_for_wmhm).get_data())
        flairs.append(nib.load(os.path.join(PATH, i, 'pre/FLAIR.nii.gz')).get_data())
        t1.append(nib.load(os.path.join(PATH, i, 'pre/T1.nii.gz')).get_data())
        clear_flairs.append(nib.load(tmp_path_to_flairs).get_data())
        
    masks = [_reshape_to(i, new_shape=(256, 256, 84)) for i in masks]
    flairs = [_reshape_to(i, new_shape=(256, 256, 84)) for i in flairs]
    t1 = [_reshape_to(i, new_shape=(256, 256, 84)) for i in t1]
    clear_flairs = [_reshape_to(i, new_shape=(256, 256, 84)) for i in clear_flairs]
    # add new axis for one channel
    masks = np.array(masks)[:, np.newaxis].astype(int)
    flairs = np.array(flairs)[:, np.newaxis]
    clear_flairs = np.array(clear_flairs)[:, np.newaxis]
    t1 = np.array(t1)[:, np.newaxis]
    # standartize data
    flair_std = flairs.std()
    flairs = flairs / flair_std
    #
    t1_std = t1.std()
    t1 = t1 / t1_std
    #
    flair_std = clear_flairs.std()
    clear_flairs = clear_flairs / flair_std
    return masks, t1, flairs, clear_flairs


def random_nonzero_crops(image: np.ndarray, mask: np.ndarray, num_of_patches=5, shape=(48, 48, 24),
                         mode='same'):
    """
    Random crops from images and it's mask over 3 last dims.
    Croped only non zero mask areas for given shape.
    
    image: array of float,
        with shape (n_batch, modalities(channel), x,y,z),
    mask: array of int,
        with shape (n_batch, 1, x,y,z),
    num_of_patches: int,
        Number of cropped patches from one image and its given mask,
    shape: int,
        Shape of crops (patches),
    mode: string,
        Type of the mode for cropping. It's can have next options:
        'same' - when shape of image crops equal to shape of the target mask,
        'half' - when shape of input image larger than shape of target mask.
    -------
    Return tuple of 2 array: crops from image, crops form mask.
    """
    patches = []
    patches_mask = []
    counter = 0
    if np.sum(mask) > 1:
        while counter < num_of_patches:
            for img, msk in zip(image, mask):
                pos = [np.random.randint(0, img.shape[j+1] - shape[j] + 1)
                       for j in range(3)]
                slices = [Ellipsis] + [slice(pos[j], pos[j] + shape[j], 1) 
                                       for j in range(3)]
                sliced_image = img[slices]
                if mode == 'half':
                    slices = [Ellipsis] + [slice(pos[j] + shape[j]//4,
                                                 pos[j] + 3*shape[j]//4, 1)
                                            for j in range(3)]
                sliced_mask = msk[slices]

                if np.sum(sliced_mask) > 15:
                    counter+=1
                    patches.append(sliced_image)
                    patches_mask.append(sliced_mask)
    else:
        raise Exception('Input contains empty mask')
    patches_mask = np.array(patches_mask)
    patches = np.array(patches)
    # only not empty masks
    idx = patches_mask.sum(axis=(1,2,3,4)) > 0
    patches = patches[idx]
    patches_mask = patches_mask[idx]
    
    return patches, patches_mask


#### augmentation

def augment(x: np.ndarray, y: np.ndarray):
    """
    Data random augmentation include scaling, rotation,
    axes flipping.
    __________________
    Say thanks to Max!
    """
    scipy.random.seed()
    
    scale = np.random.normal(1, 0.1, size=3)
    alpha, theta = np.random.normal(0, 10, size=2)
    
    def sc(x):
        shape = np.array(x.shape)
        x = zoom(x, scale, order=0)
        new_shape = np.array(x.shape)
        delta = shape - new_shape

        d_pad = np.maximum(0, delta)
        d_pad = list(zip(d_pad // 2, (d_pad + 1) // 2))
        d_slice = np.maximum(0, -delta)
        d_slice = zip(d_slice // 2, new_shape - (d_slice + 1) // 2)
        d_slice = [slice(x, y) for x, y in d_slice]

        x = x[d_slice]
        x = np.pad(x, d_pad, mode='constant')
        return x
    
    def rot(x, order=3, theta=0, alpha=0):
        x = rotate(x, theta, axes=(len(x.shape) - 2, len(x.shape) - 3), reshape=False, order=order)
        return rotate(x, alpha, axes=(len(x.shape) - 2, len(x.shape) - 1), reshape=False, order=order)
    
    for i in range(1, 4):
        if np.random.binomial(1, .5):
            x = np.flip(x, -i)
            y = np.flip(y, -i)
      
    # x = np.array([sc(i) for i in x])
    # y = sc(y[0])[np.newaxis]
    
    x = rot(x, 3, theta, alpha)
    y = rot(y, 0, theta, alpha)
    
    if np.random.binomial(1, .5):
        t = np.random.choice([-90, 0, 90])
        a = np.random.choice([-90, 0, 90])
        x = rot(x, 3, t, a)
        y = rot(y, 0, t, a)

    x = np.array([i * np.random.normal(1, 0.5) for i in x])
    return x, y


#region from_medim

def _get_steps(shape: np.ndarray, n_parts_per_axis):
    n_parts_per_axis = np.array(n_parts_per_axis)
    steps = shape // n_parts_per_axis
    steps += (shape % n_parts_per_axis) > 0
    return steps

def _build_shape(x_parts, n_parts_per_axis):
    n_dims = len(n_parts_per_axis)
    n_parts = len(x_parts)
    shape = []
    for i in range(n_dims):
        step = np.prod(n_parts_per_axis[i + 1:], dtype=int)
        s = sum([x_parts[j*step].shape[i] for j in range(n_parts_per_axis[i])])
        shape.append(s)
    return shape

def _combine_with_shape(x_parts, n_parts_per_axis, shape):
    steps = _get_steps(np.array(shape), n_parts_per_axis)
    x = np.zeros(shape, dtype=x_parts[0].dtype)
    for i, idx in enumerate(product(*map(range, n_parts_per_axis))):
        lb = np.array(idx) * steps
        slices = [*map(slice, lb, lb + steps)]
        x[slices] = x_parts[i]
    return x

def divide(x: np.ndarray, padding, n_parts_per_axis):
    """
    Divides padded x (should be padded beforehand)
    into multiple parts of about the same shape according to
    n_parts_per_axis list and padding.
    """
    padding = np.array(padding)
    steps = _get_steps(np.array(x.shape) - 2 * padding, n_parts_per_axis)
    x_parts = []
    for idx in product(*map(range, n_parts_per_axis)):
        lb = np.array(idx) * steps
        slices = [*map(slice, lb, lb + steps + 2 * padding)]
        x_parts.append(x[slices])
    return x_parts

def combine(x_parts, n_parts_per_axis):
    """
    Combines parts of one big array back into one array,
    according to n_parts_per_axis.
    """
    assert x_parts[0].ndim == len(n_parts_per_axis)
    shape = _build_shape(x_parts, n_parts_per_axis)
    x = _combine_with_shape(x_parts, n_parts_per_axis, shape)
    return x

#endregion


#region old_code

# def elastic_transform(x, alpha, sigma, axes=None):
#     """
#     Apply a gaussian elastic transform to a np.array along given axes.
#     ________
#     Say thanks to Max!
#     """
#
#     if axes is None:
#         axes = range(x.ndim)
#     axes = list(sorted(axes))
#     x = np.array(x)
#     shape = np.array(x.shape)
#     grid_shape = shape[axes]
#     shape[axes] = 1
#
#     dr = [gaussian_filter(np.random.rand(*grid_shape) * 2 - 1, sigma, mode="constant") * alpha
#           for _ in range(len(grid_shape))]
#     r = np.meshgrid(*[np.arange(i) for i in grid_shape], indexing='ij')
#
#     indices = [np.reshape(k + dk, (-1, 1)) for k, dk in zip(r, dr)]
#
#     result = np.empty_like(x)
#     for idx in np.ndindex(*shape):
#         idx = list(idx)
#         for ax in axes:
#             idx[ax] = slice(None)
#
#         z = x[idx]
#         result[idx] = map_coordinates(z, indices, order=1).reshape(z.shape)
#     return result


# def divider(X, y, with_shape=(1, 1, 40, 40, 40), mode='half'):
#     """
#     Split data into small tensors.
#     """
#     # but what if not strict size for divide?
#     print ((np.array(X.shape) % np.array(with_shape)).astype(np.int32))
#     steps = (np.array(X.shape) // np.array(with_shape)).astype(np.int32)
#     slices_targ = []
#     slices_inp = []
#     for pos in product(*map(range, steps)):
#         pos = np.array(pos) * np.array(with_shape)
#         slices_targ.append((np.array([...] +
#                                      [slice(pos[-i], pos[-i] + with_shape[-i])
#                                       for i in range(3, 0, -1)])))
#         if mode == 'half':
#             slices_inp.append((np.array([...] +
#                                     [slice(pos[-i], pos[-i] + 2 * with_shape[-i])
#                                              for i in range(3, 0, -1)])))
#
#     if mode=='half':
#         X = _reshape_to(X, np.array(X.shape) + np.array([0, 0, *np.array(with_shape[2:])]))
#         print (X.shape)
#     else:
#         slices_inp = slices_targ
#     return (np.array([X[list(slice)] for slice in slices_inp]),
#             np.array([y[list(slice)] for slice in slices_targ]),
#             slices_targ)
#
#
# def combiner(predictions, slices, shape_of_big_tensor):
#     """
#     Combine predictions into one tesor.
#     """
#     x = np.zeros(shape_of_big_tensor)
#     for i, slice in enumerate(slices):
#         x[list(slice)] = predictions[i]
#     return np.array(x)

#endregion
