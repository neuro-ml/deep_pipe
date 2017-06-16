import multiprocessing
import numpy as np
from nn_utils import iterate_minibatches
    

def __process__(q_in, q_out, function):
    np.random.seed()
    while True:
        data = q_in.get()
        if data is None:
            break
        q_out.put(function(data[0], data[1]))
    

def par_iterate_minibatches(inputs, targets, batchsize, function, shuffle=True):
    nprocs = multiprocessing.cpu_count() - 1
    q_in = multiprocessing.Queue()
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=__process__, args=(q_in, q_out, function))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    instance = iterate_minibatches(inputs, targets, batchsize, shuffle)
    batch = next(instance)
    for i in zip(*batch):
        q_in.put(i)
    last_size = len(batch[0])

    for batch in instance:
        for i in zip(*batch):
            q_in.put(i)
        data = [q_out.get() for _ in range(last_size)]
        yield list(zip(*data))
        last_size = len(batch[0])

    data = [q_out.get() for _ in range(last_size)]
    yield list(zip(*data))

    for i in range(nprocs):
        q_in.put(None)
    for p in proc:
        p.join()
