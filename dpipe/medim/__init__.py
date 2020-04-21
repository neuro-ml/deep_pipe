from warnings import warn

from . import preprocessing as prep
from . import box
from . import utils
from . import metrics
from . import patch
from . import grid
from . import shape_utils

msg = 'dpipe.medim is deprecated in favor of dpipe.im'
warn(msg, DeprecationWarning)
warn(msg, UserWarning)
