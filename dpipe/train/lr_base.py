from warnings import warn

from numpy import deprecate

from .policy import Policy

LearningRatePolicy = deprecate(Policy, old_name='LearningRatePolicy', new_name='Policy')

warn('dpipe.train.lr_base is deprecated', DeprecationWarning)
