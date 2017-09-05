import os
import json

import numpy as np
from tqdm import tqdm

from dpipe.config import get_config, get_resource_manager
from dpipe.medim.metrics import multichannel_dice_score

if __name__ == '__main__':
    resource_manager = get_resource_manager(
        get_config('config_path', 'dices_path', 'predictions_path')
    )

    dataset = resource_manager['dataset']
    dices_path = resource_manager['dices_path']
    predictions_path = resource_manager['predictions_path']

    dices = {}
    for f in tqdm(os.listdir(predictions_path)):
        identifier = f.replace('.npy', '')
        y_true = dataset.load_msegm(identifier)
        y = np.load(os.path.join(predictions_path, f))

        dices[identifier] = multichannel_dice_score(y, y_true)

    with open(dices_path, 'w') as f:
        json.dump(dices, f, indent=0)
