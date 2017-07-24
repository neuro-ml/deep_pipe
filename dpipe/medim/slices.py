def iterate_slices(*data, axis=-1, empty=True):
    """
    Iterate over slices of a series of tensors along a given axis.
    If empty is False, the last tensor in the series is assumed to be a mask.

    Parameters
    ----------
    data: list, tuple, np.array
    axis: int
    empty: bool
        whether to yield slices, containing only zeroes in the mask.
    """
    for i in range(data[0].shape[axis]):
        if empty or data[-1].take(i, axis=axis).any():
            result = [entry.take(i, axis=axis) for entry in data]
            if len(data) == 1:
                result = result[0]
            yield result
