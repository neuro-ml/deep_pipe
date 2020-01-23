Working with patches
====================

If your pipeline requires images of a given shape, you may want to split
larger images into patches, perform some operations and then combine the
results.

.. code-block:: python3

    !wget https://www.bluecross.org.uk/sites/default/files/d8/assets/images/118809lprLR.jpg

.. code-block:: python3

    import numpy as np
    from imageio import imread
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    image = imread('118809lprLR.jpg')

.. code-block:: python3

    plt.imshow(image)

Probability maps
----------------

.. code-block:: python3

    from torchvision.models import resnet50
    from torchvision.transforms import Normalize
    
    model = resnet50(pretrained=True)
    # resnet requires normalization
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

We’ll classify this image by averaging the logits on each patch. We’ll
be taking patches in a convolution-like fashion, i.e. with a fixed
stride.

.. code-block:: python3

    from dpipe.medim import grid
    from dpipe.torch import to_var, to_np
    from scipy.special import softmax

.. code-block:: python3

    from dpipe.medim.shape_utils import shape_after_convolution
    
    x = np.moveaxis(image.astype('float32'), -1, 0) # move channels forward
    x = x / 256
    
    probas = []
    for patch in grid.divide(x, patch_size=(256, 256), stride=32, valid=True):
        # move the patch to the same device as the model
        patch = to_var(patch, device=model)
        patch = normalize(patch)
        pred = to_np(model(patch[None])[0])
        pred = softmax(pred)
        
        # according to https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
        # 281 is "tabby, tabby cat"
        probas.append(pred[281][None, None])
    
    output_shape = shape_after_convolution(x.shape[1:], kernel_size=256, stride=32)
    # combine "patches" of shape (1, 1) into an image of `output_shape` with stride 1
    heatmap = grid.combine(probas, output_shape, stride=(1, 1))

.. code-block:: python3

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(heatmap)
    plt.subplot(1, 2, 2)
    plt.imshow(image)

Patches segmentation
--------------------

.. code-block:: python3

    from torchvision.models.segmentation import fcn_resnet101

.. code-block:: python3

    model = fcn_resnet101(pretrained=True)

.. code-block:: python3

    pred.shape

.. code-block:: python3

    x = np.moveaxis(image.astype('float32'), -1, 0) # move channels forward
    x = x / 256
    
    probas = []
    for patch in grid.divide(x, patch_size=(256, 256), stride=32):
        # move the patch to the same device as the model
        patch = to_var(patch, device=model)
        patch = normalize(patch)
        
        pred = model(patch[None])['out'][0]
        pred = to_np(pred)
        # 'cat' is 8
        pred = pred[8]
        
        probas.append(pred)
    
    segmentation = grid.combine(probas, x.shape[1:], stride=(32, 32))

.. code-block:: python3

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(segmentation)
    plt.subplot(1, 2, 2)
    plt.imshow(image)

Using predictors
----------------

The previous approach is a quite common pattern: split -> segment ->
combine, that’s why there is a predictor that reduces boilerplate code:

.. code-block:: python3

    from dpipe.predict import patches_grid
    
    
    @patches_grid(patch_size=(256, 256), stride=(32, 32), padding_values=None)
    def segment(patch):
        patch = to_var(patch, device=model)
        patch = normalize(patch)
        
        pred = model(patch[None])['out'][0]
        # 'cat' is 8
        return to_np(pred[8])

You can then reuse this function:

.. code-block:: python3

    segmentation = segment(image)
