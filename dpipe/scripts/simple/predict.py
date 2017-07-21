import os

import numpy as np
import tensorflow as tf

from dpipe.config import config_dataset, config_frozen_model
from dpipe.config.default_parser import get_config
from utils import read_lines

if __name__ == '__main__':
    config = get_config('ids_path', 'restore_model_path', 'predictions_path')

    predictions_path = config['predictions_path']
    ids_path = config['ids_path']
    restore_model_path = config['restore_model_path']
    dataset = config_dataset(config)
    frozen_model = config_frozen_model(config, n_chans_in=dataset.n_chans_mscan,
                                       n_chans_out=dataset.n_chans_msegm)

    ids = read_lines(ids_path)
    os.makedirs(predictions_path)

    with tf.Session(graph=frozen_model.graph) as session:
        frozen_model.prepare(session, restore_model_path)
        for identifier in ids:
            x = dataset.load_x(identifier)
            y = frozen_model.predict_object(x)

            np.save(os.path.join(predictions_path, str(identifier)), y)

            # saving some memory
            del x, y
