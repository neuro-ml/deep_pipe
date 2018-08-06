from functools import wraps, partial


def check_len(*args):
    lengths = list(map(len, args))
    if not all(length == lengths[0] for length in lengths):
        raise ValueError(f'Arguments of equal length are required: {", ".join(map(str, lengths))}')


def check_bool(*arrays):
    for i, a in enumerate(arrays):
        assert a.dtype == bool, f'{i}: {a.dtype}'


def check_shapes(*arrays):
    shapes = [array.shape for array in arrays]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError(f'Arrays of equal shape are required: {", ".join(map(str, shapes))}')


def add_check_function(func, check_function):
    """Performs a check of the function's arguments before calling it."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        check_function(*args, *kwargs.values())
        return func(*args, **kwargs)

    return wrapper


add_check_bool = partial(add_check_function, check_function=check_bool)
add_check_shapes = partial(add_check_function, check_function=check_shapes)
add_check_len = partial(add_check_function, check_function=check_len)
