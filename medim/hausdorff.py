from hausdorff import weighted_hausdorff


# Before using this module, install the dependecies:

# git clone https://github.com/mavillan/py-hausdorff.git
# pip install Cython
# cd py-hausdorff
# python setup.py build && python setup.py install


def hausdorff(a, b, weights=1, label=1):
    """
    Calculates the Hausdorff distance between two masks.
    
    Parameters
    ----------
    
    a, b: ndarray
        The arrays containing the masks. Their ndim must match.
    label: int, default = 1
        The label of the mask
    weights: number or array/list/tuple
        The weight along each axis (for anisotropic grids). If array, its length must 
        match the ndim of the array. If number, all the axes will have the same weight
    """
    
    try:
#         check if array
        len(weights)
    except TypeError:
        weights = [weights] * a.ndim 
    weights = np.array(weights, 'float64')
    
    def prep(x):
        return np.argwhere(x == label).copy(order='C').astype('float64')
    return weighted_hausdorff(prep(a), prep(b), weights)