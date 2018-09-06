from warnings import warn

from numpy import deprecate

from .policy import *

LearningRatePolicy = deprecate(Policy, old_name='LearningRatePolicy', new_name='Policy')

warn('dpipe.train.lr_policy is deprecated', DeprecationWarning)
