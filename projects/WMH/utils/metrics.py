import numpy as np
from hausdorff import weighted_hausdorff


def hausdorff(a, b, weights=1, label=1):
    """
    Calculates the Hausdorff distance between two masks.
    
    git clone https://github.com/mavillan/py-hausdorff.git
    pip install Cython
    cd py-hausdorff
    python setup.py build && python setup.py install
       
    Parameters
    ----------
    
    a, b: ndarray
       The arrays containing the masks
    label: int, default = 1
       The label of the mask
    weights: number or array/list/tuple
        The weight along each axis (for anisotropic grids). If array, its length must 
        match the ndim of the array. If number, all the axes will have the same weight   
     
    Examples
    --------
    hausdorff(x, y, weights=2) # isotropic, but weighted
    hausdorff(x, y, weights=(1,1,1,5)) # anisotropic
    """
    
    try:
        len(weights) # check if array
    except TypeError:
        weights = [weights] * a.ndim 
    weights = np.array(weights, 'float64')
    
    def prep(x):
        return np.argwhere(x == label).copy(order='C').astype('float64')
    return weighted_hausdorff(prep(a), prep(b), weights)


def dice(im1, im2, empty_score=0.):
    """
    Compute dice score.
    """
    im1 = np.asarray(im1).astype(np.bool).reshape(-1)
    im2 = np.asarray(im2).astype(np.bool).reshape(-1)
    assert im1.shape == im2.shape
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum

