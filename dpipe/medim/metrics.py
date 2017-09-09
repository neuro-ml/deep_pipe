import numpy as np


def dice_score(y_pred, target, empty_val=0):
    """Dice score between two binary masks."""
    assert y_pred.dtype == target.dtype == np.bool

    num = 2 * np.sum(y_pred & target)
    den = np.sum(y_pred) + np.sum(target)

    return empty_val if den == 0 else num / den


def multichannel_dice_score(a, b, empty_val=0):
    dices = [dice_score(x, y, empty_val=empty_val) for x, y in zip(a, b)]
    return dices


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
