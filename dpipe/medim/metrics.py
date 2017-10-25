import numpy as np

def soft_weighted_dice_score(a, b, empty_val: float = 0):
    """
    Realization of dice metric proposed in https://arxiv.org/pdf/1707.01992.pdf

    Parameters
    ----------
    a: ndarray
        Predicted probability maps
    b: ndarray of np.bool
        Ground truth
    empty_val: int
        Default value to avoid division by zero
    """
    assert b.dtype == np.bool
    assert a.shape == b.shape

    swds = 0
    num_classes = a.shape[0]

    for x, y in zip(a, b):
        num = 2 * np.sum(x * y)
        den = np.sum(y) + np.sum(x ** 2)

        swds += empty_val if den == 0 else num / den

    swds = swds / num_classes
    return swds


def dice_score(x, y, empty_val=0):
    """Dice score between two binary masks."""
    assert x.dtype == y.dtype == np.bool
    assert x.shape == y.shape

    num = 2 * np.sum(x & y)
    den = np.sum(x) + np.sum(y)

    return empty_val if den == 0 else num / den


def multichannel_dice_score(a, b, empty_val=0):
    dices = [dice_score(x, y, empty_val=empty_val) for x, y in zip(a, b)]
    return dices

def find_bounds(y):
    """

    Parameters
    ----------
    y: ndarray of np.bool
        Multichannel binary mask

    """
    assert y.dtype == np.bool

    all_bounds = []
    for channel in y:
        bounds = np.zeros(channel.shape).astype(np.bool)
        for i in range(len(channel.shape)):
            temp_bound = (channel & (~np.roll(channel, 1, axis=i))) | (channel & (~np.roll(channel, -1, axis=i)))
            bounds = bounds | temp_bound

        all_bounds.append(bounds)
    return np.array(all_bounds)

def weighted_bounds_dice_score(a,b, size_of_bound = 1, empty_val = 0):
    """

    Parameters
    ----------
    a: ndarray of np.bool
        Multichannel binary mask
    b: ndarray
        Predicted probability maps
    size_of_bound: int
    empty_val: int
        Default value to avoid division by zero
    Returns
    -------

    """

    assert b.dtype == np.bool
    assert a.shape == b.shape

    temp_b = b.copy()
    bounds = []

    for i in range(size_of_bound):
        bounds.append(find_bounds(temp_b))
        temp_b = ~np.logical_xor(temp_b, bounds[-1])

    bounds = np.array(bounds).astype(np.int) * np.arange(1,size_of_bound+1)
    temp_b = np.array(temp_b).astype(np.int)

    for bound in bounds:
        temp_b = temp_b + bound
    #now temp_b is mask with weighted boundes

    wbds = 0
    num_classes = a.shape[0]

    for x, y in zip(a, temp_b):
        num = 2 * np.sum(x * y)
        den = np.sum(y) + np.sum(x)

        wbds += empty_val if den == 0 else num / den

    wbds = wbds / num_classes
    return wbds


# Before using this module, install the dependencies:

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
        
    Examples
    --------
    hausdorff(x, y, weights=2) # isotropic, but weighted
    hausdorff(x, y, weights=(1,1,1,5)) # anisotropic
    """
    from hausdorff import weighted_hausdorff

    try:
        # check if array
        len(weights)
    except TypeError:
        weights = [weights] * a.ndim
    weights = np.array(weights, 'float64')

    def prep(x):
        return np.argwhere(x == label).copy(order='C').astype('float64')

    return weighted_hausdorff(prep(a), prep(b), weights)


def calc_max_dices(y_true, y_pred):
    dices = []
    thresholds = np.linspace(0, 1, 20)
    for true, pred in zip(y_true, y_pred):
        temp = []
        for i in range(len(true)):
            temp.append([dice_score(pred[i] > thr, true[i].astype(bool))
                         for thr in thresholds])
        dices.append(temp)
    dices = np.asarray(dices)
    return dices.mean(axis=0).max(axis=1)
