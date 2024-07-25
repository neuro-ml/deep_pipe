from dpipe.train import Switch, Exponential, Schedule


def test_switch():
    values = {1: 10, 5: 3, 7: 2, 12: 22}
    policy = Switch(0, values)
    for epoch in range(50):
        policy.epoch_started(epoch)
        if epoch in values:
            assert policy.value == values[epoch]


def test_exponential():
    policy = Exponential(1, 2, floordiv=False)
    for epoch in range(20):
        policy.epoch_started(epoch)
        assert policy.value == 2 ** epoch


def test_schedule():
    mul = {1: 10, 5: 3, 7: 2, 12: 22}
    values = {1: 10, 5: 30, 7: 60, 12: 60 * 22}
    policy = Schedule(1, mul)
    for epoch in range(50):
        policy.epoch_started(epoch)
        if epoch in values:
            assert policy.value == values[epoch]
