import numpy as np
from itertools import product


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


def check_neighborhood(y, voxel_index, interested_value: int):
    if y[voxel_index] != 1:
        return False
    voxel_index = np.asarray(voxel_index)
    values = [-1, 0, 1]
    n = y.ndim

    for delta in product(values, repeat=n):
        if sum(np.abs(delta)) == 0:
            continue
        if y[tuple(voxel_index + delta)] == interested_value:
            return True

    return False


def get_next_bound(y, bound_indexes, next_bound_value, default_value=1):
    new_bound = []
    values = [-1, 0, 1]
    n = y.ndim

    for bv in bound_indexes:
        for delta in product(values, repeat=n):
            if sum(np.abs(delta)) == 0:
                continue
            if y[tuple(bv + delta)] == default_value:
                y[tuple(bv + delta)] = next_bound_value
                new_bound.append(bv + delta)

    return new_bound


def get_weighted_mask(y, thickness=1):
    """

    Parameters
    ----------
    y: ndarray of type np.bool
        Binary mask (all dimensions must be spatial)
    thickness: int
        The thickness of the boundary
    """
    assert y.dtype == np.bool

    y = y.astype(np.int)
    bound_weight = thickness
    bw = bound_weight + 1
    bound_indexes = []

    #   firstly let's find the first bound
    #   bound arrays must contain np.arrays, not tuples
    for voxel_index in np.ndindex(*y.shape):
        if check_neighborhood(y, voxel_index, interested_value=0):
            y[voxel_index] = bw
            bound_indexes.append(np.asarray(voxel_index))

        #   other bounds
    for bw in range(bound_weight, 1, -1):
        bound_indexes = get_next_bound(y, bound_indexes, bw)

    return y


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
