Batch iterators
===============

Batch iterators are built using the following constructor:

.. code-block:: python3

    from dpipe.batch_iter import Infinite

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
    
    batch_iter = Infinite(
        load_by_random_id(dataset.load_image, dataset.load_label, ids=dataset.ids),
        batch_size=30, batches_per_epoch=10,
    )

``load_by_random_id`` infinitely yields data randomly sampled from the
dataset:

.. code-block:: python3

    for x, y in load_by_random_id(dataset.load_image, dataset.load_label, ids=dataset.ids):
        print(x.shape, y)
        break


.. parsed-literal::

    (1, 28, 28) 4


We use infinite sources because our batch iterators are executed in a
background thread, this allows us to use the resources more efficiently.

Now we can simply iterate over ``batch_iter``:

.. code-block:: python3

    # give 10 batches of size 30
    for xs, ys in batch_iter():
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
    for xs, ys in batch_iter():
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

Or you can use it as a context manager:

.. code-block:: python3

    batch_iter = Infinite(
        load_by_random_id(dataset.load_image, dataset.load_label, ids=dataset.ids),
        batch_size=30, batches_per_epoch=10,
    )

.. code-block:: python3

    with batch_iter:
        for xs, ys in batch_iter():
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


Transformations
~~~~~~~~~~~~~~~

Let’s add more transformations to the data.

.. code-block:: python3

    from dpipe.medim.shape_ops import zoom
    
    def zoom_image(pair):
        image, label = pair
        return zoom(image, scale_factor=[2, 2]), label

.. code-block:: python3

    batch_iter = Infinite(
        load_by_random_id(dataset.load_image, dataset.load_label, ids=dataset.ids), # yields pairs
        zoom_image, # zoom the images by a factor of 2
        
        batch_size=30, batches_per_epoch=3,
    )

.. code-block:: python3

    with batch_iter:
        for xs, ys in batch_iter():
            print(xs.shape, ys.shape)


.. parsed-literal::

    (30, 1, 56, 56) (30,)
    (30, 1, 56, 56) (30,)
    (30, 1, 56, 56) (30,)


Note, that because ``load_by_random_id`` yields pairs, ``pair`` is the
input of ``zoom_image``. This is not very user-friendly, that’s why
there are a number of wrappers for transformers:

.. code-block:: python3

    from dpipe.batch_iter.utils import unpack_args
    
    # a better version of zoom
    def zoom_image(image, label):
        return zoom(image, scale_factor=[2, 2]), label
    
    
    batch_iter = Infinite(
        load_by_random_id(dataset.load_image, dataset.load_label, ids=dataset.ids),
        unpack_args(zoom_image), # unpack the arguments before calling the function
        
        batch_size=30, batches_per_epoch=3)
    
    # or use a lambda directly
    batch_iter = Infinite(
        load_by_random_id(dataset.load_image, dataset.load_label, ids=dataset.ids),
        unpack_args(lambda image, label: [zoom(image, scale_factor=[2, 2]), label]),
        
        batch_size=30, batches_per_epoch=3)

However, there is still redundancy: the ``label`` argument is simply
passed through, only the ``image`` is transformed. Let’s fix that:

.. code-block:: python3

    from dpipe.batch_iter.utils import apply_at
    
    batch_iter = Infinite(
        load_by_random_id(dataset.load_image, dataset.load_label, ids=dataset.ids),
        apply_at(0, zoom, scale_factor=[2, 2]),
        
        batch_size=30, batches_per_epoch=3)

.. code-block:: python3

    with batch_iter:
        for xs, ys in batch_iter():
            print(xs.shape, ys.shape)


.. parsed-literal::

    (30, 1, 56, 56) (30,)
    (30, 1, 56, 56) (30,)
    (30, 1, 56, 56) (30,)


Now we don’t even have to create another function!

Check ``dpipe.batch_iter.utils`` for other helper functions.
