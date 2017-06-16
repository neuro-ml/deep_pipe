import numpy as np


def iterate_minibatches(inputs, targets, 
                        batch_size, shuffle=True, strict=True):
    """
    Classical batch iterator,
    """
    assert len(inputs) == len(targets)
    last = batch_size - 1 if strict else 0
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - last, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
