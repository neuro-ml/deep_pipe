from functools import wraps


def check_len(*arrays):
    length = len(arrays[0])
    for i, a in enumerate(arrays):
        assert length == len(a), f'Different len: {arrays}'


def check_bool(*arrays):
    for i, a in enumerate(arrays):
        assert a.dtype == bool, f'{i}: {a.dtype}'


def check_shapes(*arrays):
    shapes = [array.shape for array in arrays]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError(f'Arrays of equal shape are required: {", ".join(map(str, shapes))}')


def add_check_bool(func):
    """Check that all function arguments are boolean arrays."""

    @wraps(func)
    def new_func(*args, **kwargs):
        check_bool(*args, *kwargs.values())
        return func(*args, **kwargs)

    return new_func


def add_check_shapes(func):
    """Check that all function arguments are arrays with equal shape."""

    @wraps(func)
    def new_func(*args, **kwargs):
        check_shapes(*args, *kwargs.values())
        return func(*args, **kwargs)

    return new_func
