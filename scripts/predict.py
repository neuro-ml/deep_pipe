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

    frozen_model = rm.frozen_model
    batch_predict = rm.batch_predict
    for identifier in tqdm(rm.ids):
        x = rm.load_x(identifier)
        y = batch_predict.predict(x, predict_fn=frozen_model.do_inf_step)

        np.save(os.path.join(rm.predictions_path, str(identifier)), y)
        # saving some memory
        del x, y
