import os
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from dpipe.config import get_config, get_resource_manager

if __name__ == '__main__':
    resource_manager = get_resource_manager(get_config(
        'config_path', 'ids_path', 'restore_model_path', 'predictions_path'
    ))

    predictions_path = resource_manager['predictions_path']
    restore_model_path = resource_manager['restore_model_path']
    ids = resource_manager['ids']
    load_x = resource_manager['load_x']

    frozen_model = resource_manager['frozen_model']

    os.makedirs(predictions_path)

    with tf.Session(graph=frozen_model.graph) as session:
        frozen_model.prepare(session, restore_model_path)
        for identifier in tqdm(ids):
            x = load_x(identifier)
            y = frozen_model.predict_object(x)

            np.save(os.path.join(predictions_path, str(identifier)), y)
            # saving some memory
            del x, y
