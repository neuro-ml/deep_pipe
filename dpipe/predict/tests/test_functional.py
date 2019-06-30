from dpipe.predict.functional import *


def test_chain_decorators():
    def append(num):
        def decorator(func):
            def wrapper():
                return func() + [num]

            return wrapper

        return decorator

    @append(1)
    @append(2)
    @append(3)
    def f():
        return []

    chained = chain_decorators(
        append(1), append(2), append(3),
        predict=lambda: []
    )

    assert f() == chained()
