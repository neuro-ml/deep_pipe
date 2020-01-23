Predict
=======

Usually when dealing with neural networks, at inference time the input
data may require some preprocessing before being fed into the network.
Also, the network’s output might need postprocessing in order to obtain
a final prediction.

Padding and cropping
~~~~~~~~~~~~~~~~~~~~

Let’s suppose that we have a ``network`` for segmentation that can only
work with images larger than 256x256 pixels.

Before feeding a given ``image`` into the network you may want to pad
it:

.. code-block:: python3

    from dpipe.medim.shape_ops import pad_to_shape
    
    padded = pad_to_shape(image, np.maximum(image.shape, (256, 256)))
    mask = network(padded)

Now you need to remove the padding in order to make the ``mask`` of same
shape as ``image``:

.. code-block:: python3

    from dpipe.medim.shape_ops import crop_to_shape
    
    mask = crop_to_shape(mask, image.shape)

Let’s make a function that implements the whole pipeline:

.. code-block:: python3

    import numpy as np
    from dpipe.medim.shape_ops import pad_to_shape, crop_to_shape
    
    def predict_pad(image, network, min_shape):
        # pad
        padded = pad_to_shape(image, np.maximum(image.shape, min_shape))
        # predict
        mask = network(padded)
        # restore
        mask = crop_to_shape(mask, image.shape)
        return mask

Now we have a perfectly reusable function.

Scale
~~~~~

Now let’s write a function that downsamples the input by a factor of 2
and then zooms the output by 2.

.. code-block:: python3

    import numpy as np
    from dpipe.medim.shape_ops import zoom, zoom_to_shape
    
    def predict_zoom(image, network, scale_factor=0.5):
        # zoom
        zoomed = zoom(image, scale_factor)
        # predict
        mask = network(zoomed)
        # restore
        mask = zoom_to_shape(mask, image.shape)
        return mask

Combining
~~~~~~~~~

Now suppose we want to combine zooming and padding. We could do
something like:

.. code-block:: python3

    import numpy as np
    from dpipe.medim.shape_ops import pad_to_shape, crop_to_shape
    
    def predict(image, network, min_shape, scale_factor):
        # zoom
        zoomed = zoom(image, scale_factor)
        
        # ---
        # pad
        padded = pad_to_shape(image, np.maximum(zoomed.shape, min_shape))
        # predict
        mask = network(padded)
        # restore
        mask = crop_to_shape(mask, np.minimum(mask.shape, zoomed.shape))
        # ---
        
        mask = zoom_to_shape(mask, image.shape)
        return mask

Note how the content of ``predict`` is divided in two regions: basically
it looks like the function ``predict_zoom`` but with the line

::

   mask = network(padded)

replaced by the body of ``predict_pad``.

Basically, it means that we can pass ``predict_pad`` as the ``network``
argument and reuse the functions we defined above:

.. code-block:: python3

    def predict(image, network, min_shape, scale_factor):
        def network_(x):
            return predict_pad(x, network, min_shape)
        
        return predict_zoom(image, network_, scale_factor)

``predict_pad`` “wraps” the original ``network`` - it behaves like
``network``, and ``predict_zoom`` doesn’t really care whether it
received the original ``network`` or a wrapped one.

This sounds just like a decorator (a very good explanation can be found
`here <https://stackoverflow.com/questions/739654/how-to-make-a-chain-of-function-decorators/1594484#1594484>`__).

If we implement ``predict_pad`` and ``predict_zoom`` as decorators we
can more easily reuse them:

.. code-block:: python3

    def predict_pad(min_shape):
        def decorator(network):
            def predict(image):
                # pad
                padded = pad_to_shape(image, np.maximum(image.shape, min_shape))
                # predict
                mask = network(padded)
                # restore
                mask = crop_to_shape(mask, np.minimum(mask.shape, image.shape))
                return mask
            
            return predict
        return decorator
    
    def predict_zoom(scale_factor):
        def decorator(network):
            def predict(image):
                # zoom
                zoomed = zoom(image, scale_factor)
                # predict
                mask = network(padded)
                # restore
                mask = zoom_to_shape(mask, image.shape)
                return mask
    
            return predict
        return decorator

Then the same ``predict`` can be defined like so:

.. code-block:: python3

    @predict_zoom(0.5)
    @predict_pad((256, 256))
    def predict(image):
        # here the image is already zoomed and padded
        return network(image)

Now ``predict`` is just a function that receives a single argument - the
image.

If you don’t like the decorator approach you can use a handy function
for that:

.. code-block:: python3

    from dpipe.predict.functional import chain_decorators
    
    predict = chain_decorators(
        predict_zoom(0.5), 
        predict_pad((256, 256)),
        predict=network,
    )

which gives the same function.
