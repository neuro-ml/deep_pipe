import os
import json

import numpy as np
from tqdm import tqdm

from dpipe.config import get_args, get_resource_manager
from dpipe.medim.metrics import multichannel_dice_score

if __name__ == '__main__':
    rm = get_resource_manager(
        **get_args('config_path', 'dices_path', 'predictions_path')
    )

    dices = {}
    for f in tqdm(os.listdir(rm.predictions_path)):
        identifier = f.replace('.npy', '')
        y_true = rm.dataset.load_msegm(identifier)
        y = np.load(os.path.join(rm.predictions_path, f))

        dices[identifier] = multichannel_dice_score(y, y_true)

    with open(rm.dices_path, 'w') as f:
        json.dump(dices, f, indent=0)
