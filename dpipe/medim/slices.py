from collections import deque
from typing import Union, List

import numpy as np


def join_result(stacks, axis, concatenate):
    func = np.stack
    # if only one slice then, probably, the axis is not needed
    if len(stacks[0]) == 1 and concatenate is None:
        result = [stack[0] for stack in stacks]
    else:
        if concatenate is not None:
            func = np.concatenate
            axis = concatenate
        result = [func(stack, axis) for stack in stacks]
    if len(result) == 1:
        result = result[0]
    return result


def iterate_slices(*data, axis: int = -1, slices: Union[int, List[int]] = 1,
                   pad: Union[int, List[int]] = 0, concatenate: int = None):
    """
    Iterate over slices of a series of tensors along a given axis.

    ----------
    data: Sequence
    axis: int, optional
    slices: int, List[int], optional
        The number of slices to yield
    pad: int, List[int], optional
        Padding along the specified axis
    concatenate : int, optional
        If specified, the slices will be concatenated instead of stacked
    """
    if type(slices) is int:
        slices = [slices] * len(data)
    slices = np.asarray(slices)
    if not (slices > 0).all():
        return

    if type(pad) is int:
        pad = [pad] * len(data)
    assert len(slices) == len(data) and len(pad) == len(data)

    # padding
    # TODO: memory-efficient
    new_data = []
    for pad_, entry in zip(pad, data):
        axis_ = entry.ndim + axis if axis < 0 else axis
        pads = [(pad_, pad_) if dim == axis_ else (0, 0)
                for dim in range(entry.ndim)]
        new_data.append(np.pad(entry, pads, mode='constant'))
    data = new_data
    shapes = [entry.shape[axis] for entry in data]
    assert (slices <= shapes).all()

    stacks = [deque() for _ in range(len(slices))]
    indices = np.zeros(len(data))

    # handle different number of slices
    iterate = True
    while iterate:
        iterate = False
        for i, (stack, entry, slice_) in enumerate(zip(stacks, data, slices)):
            if len(stack) < slice_:
                stack.append(entry.take(indices[i], axis=axis))
                indices[i] += 1
                iterate = True

    yield join_result(stacks, axis, concatenate)

    # yield main body
    while (indices < shapes).all():
        for stack in stacks:
            stack.popleft()

        for i, (stack, entry, slice_) in enumerate(zip(stacks, data, slices)):
            stack.append(entry.take(indices[i], axis=axis))
            indices[i] += 1
        yield join_result(stacks, axis, concatenate)
