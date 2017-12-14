Batch predictors
================

Batch predictors implement the following logic:
    1. deconstruct the object into batches
    2. feed the batches into the network
    3. In case of validation - aggregate the validation loss
    4. Build the network's prediction based on the predicted batches

The interface
-------------

.. automodule:: dpipe.batch_predict.base
    :members:
    :undoc-members:
    :show-inheritance:

dpipe\.batch\_predict\.patch\_3d module
---------------------------------------

.. automodule:: dpipe.batch_predict.patch_3d
    :members:
    :undoc-members:
    :show-inheritance:

dpipe\.batch\_predict\.patch\_3d\_fixed module
----------------------------------------------

.. automodule:: dpipe.batch_predict.patch_3d_fixed
    :members:
    :undoc-members:
    :show-inheritance:

dpipe\.batch\_predict\.simple module
------------------------------------

.. automodule:: dpipe.batch_predict.simple
    :members:
    :undoc-members:
    :show-inheritance:

dpipe\.batch\_predict\.slice module
-----------------------------------

.. automodule:: dpipe.batch_predict.slice
    :members:
    :undoc-members:
    :show-inheritance:

dpipe\.batch\_predict\.utils module
-----------------------------------

.. automodule:: dpipe.batch_predict.utils
    :members:
    :undoc-members:
    :show-inheritance:
