from warnings import warn

warn('dpipe.train.batch_iter has been moved to dpipe.batch_iter.base', DeprecationWarning)

from dpipe.batch_iter.base import *
