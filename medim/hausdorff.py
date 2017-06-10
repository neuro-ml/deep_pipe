from hausdorff import hausdorff as _hausdorff


# Before using this module, install the dependecies:

# git clone https://github.com/mavillan/py-hausdorff.git
# pip install Cython
# cd py-hausdorff
# python setup.py build && python setup.py install


def hausdorff(a, b, label=1):
    """
    Calculates the Hausdorff distance between two masks.
    
    Parameters
    ----------
    
    a, b: ndarray
       The arrays containing the masks
    label: int, default = 1
       The label of the mask
    """
    
    def prep(x):
        return np.argwhere(x == label).copy(order='C').astype('float64')
    return _hausdorff(prep(a), prep(b))