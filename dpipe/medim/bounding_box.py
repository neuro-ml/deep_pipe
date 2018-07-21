import numpy as np

from dpipe.medim.box import mask2bounding_box

get_start_stop = np.deprecate(mask2bounding_box, old_name='get_start_stop', new_name='mask2bounding_box',
                              message='Use medim.box.mask2bounding_box instead.')

