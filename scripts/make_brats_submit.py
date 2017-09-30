import os

import numpy as np
import nibabel as nib
from tqdm import tqdm

from dpipe.config import get_args

if __name__ == '__main__':
    config = get_args('predictions_path', 'submission_path')
    predictions_path = config['predictions_path']

    for filename in tqdm(os.listdir(predictions_path)):
        y_pred_proba = np.load(os.path.join(predictions_path, filename))
        y_pred = np.argmax(y_pred_proba, axis=0)
        # That was the original mask
        y_pred[y_pred == 3] = 4
        y_pred = y_pred.astype(np.uint8)
        img = nib.Nifti1Image(y_pred, np.eye(4))

        # Don't know if it's needed
        img.header.get_xyzt_units()

        new_filename = os.path.join(config['submission_path'],
                                    filename.split('.')[0] + '.nii.gz')
        nib.save(img, new_filename)
