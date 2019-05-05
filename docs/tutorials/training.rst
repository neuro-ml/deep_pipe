
Training
========

``deep_pipe`` has a unified interface for training models. We will show
an example for a model written in PyTorch:

.. code-block:: python3

    from dpipe.torch import train_model

this is the main function; it requires a `TorchModel`, a `BatchIter` ,
the learning rate and the number of epochs.

Let’s build all the required components.

Model
~~~~~

The model encapsulates all the logic necessary for training and
inference with a given architecture:

.. code-block:: python3

    from dpipe.torch import TorchModel
    from dpipe import layers
    import torch
    from torch import nn
    
    
    architecture = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3),
        
        layers.PyramidPooling(nn.functional.max_pool2d), # global pooling
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    
    model = TorchModel(architecture, nn.Softmax(-1), nn.CrossEntropyLoss(), torch.optim.Adam)

It requires the architecture itself, as well as the loss function, the
activation layer, and the optimizer class.

Batch iterator
~~~~~~~~~~~~~~

The batch iterators are covered in a separate tutorial
(:doc:`tutorials/batch_iter`), we’ll reuse the code from it:

.. code-block:: python3

    from dpipe.batch_iter import Infinite
    from dpipe.tests.mnist.resources import MNIST
    from dpipe.batch_iter import load_by_random_id
    
    dataset = MNIST('PATH TO DATA')
    
    batch_iter = Infinite(
        load_by_random_id(dataset.load_image, dataset.load_label, ids=dataset.ids),
        batch_size=100, n_iters_per_epoch=2000,
    )

Training the model
~~~~~~~~~~~~~~~~~~

Next, we just run the function ``train_model``:

.. code-block:: python3

    train_model(model, batch_iter, n_epochs=10, lr=1e-3)

Logging and Checkpoints
~~~~~~~~~~~~~~~~~~~~~~~

After the calling ``train_model`` the interpreter just “hangs” until the
training is over. In order to log various information about the training
process, you can pass the ``log_path`` argument. The logs will be
written in a format, readable by tensorboard.

Also, it is often useful to keep checkpoints (or snapshots) of you model
and optimizer in case you may want to resotore them. To do that, pass
the ``checkpoints_path`` argument:

.. code-block:: python3

    train_model(model, batch_iter, n_epochs=10, lr=1e-3,
                log_path='~/my_new_model/logs', checkpoints_path='~/my_new_model/checkpoints')

The cool part is that if the training is prematurely stopped, e.g. by an
exception, you can resume the training from the same point instead of
starting over:

.. code-block:: python3

    train_model(model, batch_iter, n_epochs=10, lr=1e-3,
                log_path='~/my_new_model/logs', checkpoints_path='~/my_new_model/checkpoints')
    # ... something bad happened, e.g. KeyboardInterrupt
    
    # start from where you left off
    train_model(model, batch_iter, n_epochs=10, lr=1e-3,
                log_path='~/my_new_model/logs', checkpoints_path='~/my_new_model/checkpoints')

Policies
~~~~~~~~

You can further customize the training process by passing addtitional
policies.

The `Policy` interface has two methods:

.. code-block:: python3

    class Policy:
        def epoch_started(self, epoch: int):
            # ...
    
        def epoch_finished(self, epoch: int, *, train_losses: Sequence = None, metrics: dict = None):
            # ...

which are called at the start and end (respectively) of each epoch. The
simplest setting in which they can be used is early stopping:

.. code-block:: python3

    import numpy as np
    from dpipe.train.policy import EarlyStopping, Policy
    
    class TrainLossStop(Policy):
        def epoch_finished(self, epoch: int, *, train_losses, metrics: dict = None):
            if np.mean(train_losses) < 1e-5:
                raise EarlyStopping

this policy raises a specific exception - `EarlyStopping`, in order to
trigger early stopping if the train loss becomes smaller than 1e-5.

Policies with values
^^^^^^^^^^^^^^^^^^^^

Another useful kind of policies are the ones that carry a value which
changes over time. For example - `Exponential`, which implements an
exponentially changing policy.

Their ``value`` attribute is passed to the model’s `do_train_step`
method as a keyword argument.

Let’s write a dummy model in order to demonstrate this.

.. code-block:: python3

    from dpipe.train.policy import Exponential
    
    
    class WeightedModel(TorchModel):
        def do_train_step(self, *inputs, lr, weights):
            print('I received these weights:', weights)
            # do something with the weights ...
            # perform a train step ...
            loss = super().do_train_step(*inputs, lr=lr)
            return loss
        
    
    model = WeightedModel(architecture, nn.Softmax(-1), nn.CrossEntropyLoss(), torch.optim.Adam)
    
    train_model(model, batch_iter, n_epochs=10, lr=1e-3, weights=Exponential(initial=10, multiplier=0.5))

.. code-block:: python3

    I received these weights: 10.0
    I received these weights: 10.0
    ...
    I received these weights: 5.0
    I received these weights: 5.0
    ...
    I received these weights: 2.5
    ...

Validation
~~~~~~~~~~

Finally, you may want to evaluate your network on a separate validation
set after each epoch. This is done by the ``validate`` argument. It
expects a function that simply returns a dictionary with the calculated
metrics, e.g.:

.. code-block:: python3

    def validate():
        return {
            'acuracy': 0.95,
            'roc_auc': 0.91,
        }

this is a dummy function, of course.
