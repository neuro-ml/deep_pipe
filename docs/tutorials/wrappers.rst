Wrappers
========

Consider the following dataset, which is a simple loader for MNIST:

.. code-block:: python3

    class MNIST:
        # ...    
        
        def load_image(self, identifier: str):
            return self.xs[int(identifier)]
    
        def load_label(self, identifier: str):
            return self.ys[int(identifier)]
        
    # The full implementation can be found at `dpipe.tests.mnist.resources`:
    # from dpipe.tests.mnist.resources import MNIST
    
    dataset = MNIST('PATH TO DATA')
    dataset.load_image(0).shape, dataset.load_label(0)




.. parsed-literal::

    ((1, 28, 28), 5)



Next, suppose you want to upsample the images by a factor of 2.

There are several solutions:

-  Rewrite the dataset - breaks compatibility, not reusable

-  Write a new dataset - not reusable, generates a lot of repetitive
   code

-  Subclass the dataset - not reusable

-  Wrap the dataset

Wrappers are handy when you need to change the dataset's behaviour in a
reusable way.

You can think of a wrapper as an additional layer around the original
dataset. In case of upsampling it could look something like this:

.. code-block:: python3

    from dpipe.dataset.wrappers import Proxy
    from dpipe.medim.shape_ops import zoom
    
    class UpsampleWrapper(Proxy):
        def load_image(self, identifier):
            # self._shadowed is the original dataset
            image = self._shadowed.load_image(identifier)
            image = zoom(image, [2, 2])
            return image

.. code-block:: python3

    upsampled = UpsampleWrapper(dataset)
    upsampled.load_image(0).shape, upsampled.load_label(0)




.. parsed-literal::

    ((1, 56, 56), 5)



Now this wrapper can be reused with other datasets that have the
``load_image`` method. Note that ``load_label`` is also working, even
though it wasn't defined in the wrapper.

``dpipe`` already has a collection of predefined wrappers, for example,
you can apply upsampling as follows:

.. code-block:: python3

    from dpipe.dataset.wrappers import apply
    
    upsampled = apply(dataset, load_image=lambda image: zoom(image, [2, 2]))

or in a more functional fashion:

.. code-block:: python3

    from functools import partial
    
    upsampled = apply(dataset, load_image=partial(zoom, scale_factor=[2, 2]))
