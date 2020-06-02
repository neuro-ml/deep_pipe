
Batch iterators
===============

Batch iterators are built using the following constructor:

.. code-block:: python3

    from dpipe.batch_iter import Infinite

its only required argument is ``source`` - an infinite iterable that
yields entries from your data.

We'll build an example batch iterator that yields batches from the MNIST
dataset:

.. code-block:: python3

    from torchvision.datasets import MNIST
    from pathlib import Path
    import numpy as np
    
    
    # download to ~/tests/MNIST, if necessary
    dataset = MNIST(Path('~/tests/MNIST').expanduser(), transform=np.array, download=True)

Sampling
~~~~~~~~

.. code-block:: python3

    from dpipe.batch_iter import sample
    
    # yield 10 batches of size 30 each epoch:
    
    batch_iter = Infinite(
        sample(dataset), # randomly sample from the dataset
        batch_size=30, batches_per_epoch=10,
    )

``sample`` infinitely yields data randomly sampled from the dataset:

.. code-block:: python3

    for x, y in sample(dataset):
        print(x.shape, y)
        break


.. parsed-literal::

    (28, 28) 7


We use infinite sources because our batch iterators are executed in a
background thread, this allows us to use the resources more efficiently.
For example, a new batch can be prepared while the network's forward and
backward passes are performed in the main thread.

Now we can simply iterate over ``batch_iter``:

.. code-block:: python3

    # give 10 batches of size 30
    for xs, ys in batch_iter():
        print(xs.shape, ys.shape)


.. parsed-literal::

    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)


... and reuse it again:

.. code-block:: python3

    # give another 10 batches of size 30
    for xs, ys in batch_iter():
        print(xs.shape, ys.shape)


.. parsed-literal::

    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)


After the training is over you must close the batch iterator in order to
stop all the background processes:

.. code-block:: python3

    batch_iter.close()

Or you can use it as a context manager:

.. code-block:: python3

    batch_iter = Infinite(
        sample(dataset),
        batch_size=30, batches_per_epoch=10,
    )
    
    with batch_iter:
        for xs, ys in batch_iter():
            print(xs.shape, ys.shape)


.. parsed-literal::

    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)
    (30, 28, 28) (30,)


Transformations
~~~~~~~~~~~~~~~

Let's add more transformations to the data.

.. code-block:: python3

    from dpipe.im import zoom
    
    def zoom_image(pair):
        image, label = pair
        return zoom(image, scale_factor=[2, 2]), label

.. code-block:: python3

    batch_iter = Infinite(
        sample(dataset), # yields pairs
        zoom_image, # zoom the images by a factor of 2
        
        batch_size=30, batches_per_epoch=3,
    )

You can think of `Infinite` as a pipe through which the data flows.

Each function takes as input the data (an ``[image, label]`` pair in
this case) applies a trasformation, and the result is propagated
further.

.. code-block:: python3

    with batch_iter:
        for xs, ys in batch_iter():
            print(xs.shape, ys.shape)


.. parsed-literal::

    (30, 56, 56) (30,)
    (30, 56, 56) (30,)
    (30, 56, 56) (30,)


Note, that because ``sample`` yields pairs, ``pair`` is the input of
``zoom_image``. This is not very user-friendly, that's why there are a
number of wrappers for transformers:

.. code-block:: python3

    from dpipe.batch_iter import unpack_args
    
    # a better version of zoom
    def zoom_image(image, label):
        return zoom(image, scale_factor=[2, 2]), label
    
    
    batch_iter = Infinite(
        sample(dataset),
        unpack_args(zoom_image), # unpack the arguments before calling the function
        
        batch_size=30, batches_per_epoch=3)
    
    # or use a lambda directly
    batch_iter = Infinite(
        sample(dataset),
        unpack_args(lambda image, label: [zoom(image, scale_factor=[2, 2]), label]),
        
        batch_size=30, batches_per_epoch=3)

However, there is still redundancy: the ``label`` argument is simply
passed through, only the ``image`` is transformed. Let's fix that:

.. code-block:: python3

    from dpipe.batch_iter import apply_at
    
    batch_iter = Infinite(
        sample(dataset),
        # apply zoom at index 0 of the pair with scale_factor=[2, 2] as an additional argument
        apply_at(0, zoom, scale_factor=[2, 2]),
        
        batch_size=30, batches_per_epoch=3)

.. code-block:: python3

    with batch_iter:
        for xs, ys in batch_iter():
            print(xs.shape, ys.shape)


.. parsed-literal::

    (30, 56, 56) (30,)
    (30, 56, 56) (30,)
    (30, 56, 56) (30,)


Now we don't even have to create another function!

Check ``dpipe.batch_iter.utils`` for other helper functions.

Parallel execution
~~~~~~~~~~~~~~~~~~

The batch iterator supports both thread-based and process-based
execution.

Threads
^^^^^^^

Wrap the function in `Threads` in order to enable thread-based
parallelism:

.. code-block:: python3

    %%time
    
    import time
    import itertools
    from dpipe.batch_iter import Threads
    
    
    def do_stuff(x):
        time.sleep(1)
        return x ** 2,
    
    batch_iter = Infinite(
        range(10),
        do_stuff, # sleep for 10 seconds
        batch_size=10, batches_per_epoch=1
    )
    
    for value in batch_iter():
        pass


.. parsed-literal::

    CPU times: user 33.3 ms, sys: 9.17 ms, total: 42.5 ms
    Wall time: 10 s


.. code-block:: python3

    %%time
    
    batch_iter = Infinite(
        range(10),
        Threads(do_stuff, n_workers=2), # sleep for 5 seconds
        batch_size=10, batches_per_epoch=1
    )
    
    for value in batch_iter():
        pass


.. parsed-literal::

    CPU times: user 21.4 ms, sys: 7.75 ms, total: 29.1 ms
    Wall time: 5.01 s


Processes
^^^^^^^^^

Similarly, wrap the function in `Loky` in order to enable process-based
parallelism:

.. code-block:: python3

    from dpipe.batch_iter import Loky

.. code-block:: python3

    %%time
    
    batch_iter = Infinite(
        range(10),
        Loky(do_stuff, n_workers=2), # sleep for 5 seconds
        batch_size=10, batches_per_epoch=1
    )
    
    for value in batch_iter():
        pass


.. parsed-literal::

    CPU times: user 43.6 ms, sys: 27.6 ms, total: 71.2 ms
    Wall time: 5.56 s


Combining objects into batches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your dataset contains items of various shapes, you can't just stack
them into batches. For example you may want to pad them to a common
shape. To do this, pass a custom ``combiner`` to `Infinite`:

.. code-block:: python3

    # random 3D images of random shapes:
    
    images = [np.random.randn(10, 10, np.random.randint(2, 40)) for _ in range(100)]
    labels = np.random.randint(0, 2, size=30)

.. code-block:: python3

    images[0].shape, images[1].shape




.. parsed-literal::

    ((10, 10, 34), (10, 10, 34))



.. code-block:: python3

    from dpipe.batch_iter import combine_pad
    
    batch_iter = Infinite(
        sample(list(zip(images, labels))),
        batch_size=5, batches_per_epoch=3, 
    #     pad and combine
        combiner=combine_pad
    )
    
    with batch_iter:
        for xs, ys in batch_iter():
            print(xs.shape, ys.shape)


.. parsed-literal::

    (5, 10, 10, 39) (5,)
    (5, 10, 10, 34) (5,)
    (5, 10, 10, 39) (5,)


Adaptive batch size
~~~~~~~~~~~~~~~~~~~

If samples in your pipeline have various sizes, a constant batch size
can be too wasteful.

You can pass a function to ``batch_size`` instead of an integer.

Let's say we are classifying 3D images of different shapes along the
last axis. We want a batch to contain at most 100 slices along the last
axis.

.. code-block:: python3

    def should_add(seq, item):
        # seq - sequence of already added objects to the batch
        # item - the next item
        
        count = 0
        for image, label in seq + [item]:
            count += image.shape[-1]
            
        return count <= 100

.. code-block:: python3

    from dpipe.batch_iter import combine_pad
    
    batch_iter = Infinite(
        sample(list(zip(images, labels))),
        
        batch_size=should_add, batches_per_epoch=3, 
        combiner=combine_pad
    )
    
    with batch_iter:
        for xs, ys in batch_iter():
            print(xs.shape, ys.shape)


.. parsed-literal::

    (5, 10, 10, 34) (5,)
    (4, 10, 10, 25) (4,)
    (4, 10, 10, 32) (4,)


Note that the batch sizes are different: 4, 4, 5
