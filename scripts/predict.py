import os
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from dpipe.config import get_args, get_resource_manager

if __name__ == '__main__':
    rm = get_resource_manager(**get_args(
        'config_path', 'ids_path', 'restore_model_path', 'predictions_path'
    ))

    os.makedirs(rm.predictions_path)

    with tf.Session(graph=rm.frozen_model.graph) as session:
        rm.frozen_model.prepare(session, rm.restore_model_path)
        for identifier in tqdm(rm.ids):
            x = rm.load_x(identifier)
            y = rm.frozen_model.predict_object(x)

            np.save(os.path.join(rm.predictions_path, str(identifier)), y)
            # saving some memory
            del x, y
