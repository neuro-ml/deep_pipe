
Batch iterators
===============

Batch iterators are buil using the following function:

.. code-block:: python3

    from dpipe.batch_iter import make_infinite_batch_iter

its only required argument is ``source`` - an infinite iterable that
yields entries from your data.

We’ll build an example batch iterator that yields batches from the MNIST
dataset:

.. code-block:: python3

    from dpipe.tests.mnist.resources import MNIST
    
    dataset = MNIST('PATH TO DATA')

.. code-block:: python3

    from dpipe.batch_iter import load_by_random_id
    
    # yield 10 batches of size 30 each epoch:
    
    batch_iter = make_infinite_batch_iter(
        load_by_random_id(dataset.load_image, dataset.load_label, ids=dataset.ids),
        batch_size=30, n_iters_per_epoch=10,
    )

``load_by_random_id`` infinitely yields data randomly sampled from the
dataset:

.. code-block:: python3

    x, y = next(load_by_random_id(dataset.load_image, dataset.load_label, ids=dataset.ids))
    x.shape, y




.. parsed-literal::

    ((1, 28, 28), 2)



We use infinite sources because our batch iterators are executed in a
background thread, this allows us to use the resources more efficiently.

Now we can simply iterate over ``batch_iter``:

.. code-block:: python3

    # give 10 batches of size 30
    for xs, ys in batch_iter:
        print(xs.shape, ys.shape)


.. parsed-literal::

    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)


… and reuse it again:

.. code-block:: python3

    # give another 10 batches of size 30
    for xs, ys in batch_iter:
        print(xs.shape, ys.shape)


.. parsed-literal::

    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)
    (30, 1, 28, 28) (30,)


After the training is over you must close the batch iterator in order to
stop all the background processes:

.. code-block:: python3

    batch_iter.close()

Or you use it as a context manager:

.. code-block:: python3

    with batch_iter:
        for xs, ys in batch_iter:
            print(xs.shape, ys.shape)
