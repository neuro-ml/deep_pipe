import os
import functools

import numpy as np
import nibabel as nib

from .base import Dataset


def cached_property(f):
    return property(functools.lru_cache(1)(f))


class WhiteMatterHyperintensity(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.idx_to_path = self._get_idx_to_paths(data_path)
        self._patient_ids = list(self.idx_to_path.keys())

    def _get_idx_to_paths(self, path):
        idx_paths = {}
        for path_, dirs, files in os.walk(path):
            if 'pre' in dirs:
                idx = path_.split('/')[-1]
                idx_paths[idx] = path_
        return idx_paths

    def _reshape_to(self, tmp: np.ndarray, new_shape = None):
        """
        Reshape ND array to new shapes.
        Parameters
        ----------
        tmp : np.array
            ND array with shapes less than new_shape.
        new_shape : tuple
            Tuple from N number - new shape.
        Returns
        ------
        result : np.array
            Return np.array with shapes equal to new_shape.
         Example.
        _ = _reshape_to(X_test[..., :80], (15, 2, 288, 288, 100))
        """
        assert not new_shape is None
        new_diff = [((new_shape[-i] - tmp.shape[-i]) // 2,
                     (new_shape[-i] - tmp.shape[-i]) // 2 + \
                        (new_shape[-i] - tmp.shape[-i]) % 2)
                    for i in range(len(new_shape), 0, -1)]
        return np.pad(tmp, new_diff, mode='constant', constant_values=0)

    def load_mscan(self, patient_id):
        path_to_modalities = self.idx_to_path[patient_id]
        res = []
        for modalities in ['pre/FLAIR.nii.gz', 'pre/T1.nii.gz']:
            image = os.path.join(path_to_modalities, modalities)
            x = nib.load(image).get_data().astype('float32')
            # add skull stripping
            if modalities == 'pre/FLAIR.nii.gz':
                brain = path_to_modalities + '/pre/brainmask_T1_mask.nii.gz'
                mask = nib.load(brain).get_data()
                x[mask == 0] = 0
            #x = self._reshape_to(x, new_shape=self.spatial_size)
            img_std = x.std()
            x = x / img_std
            res.append(x)
        return np.asarray(res)

    def load_segm(self, patient_id):
        path_to_modalities = self.idx_to_path[patient_id]
        x = nib.load(os.path.join(path_to_modalities, 'wmh.nii.gz')).get_data()
        #x = self._reshape_to(x, new_shape=self.spatial_size)
        x[x == 2] = 2
        print(np.max(x))
        return np.array(x, dtype=int)

    @property
    def patient_ids(self):
        return self._patient_ids

    @property
    def n_chans_mscan(self):
        return 2

    @cached_property
    def segm2msegm(self) -> np.array:
        """2d matrix, filled with mapping segmentation to msegmentation.
        Rows for int value from segmentation and column for channel values in
        multimodal segmentation, corresponding for each row."""
        return np.array([
            [0],
            [1],
            [0]
        ], dtype=bool)

# [[0, 0],
#  [0, 1],
#  [1, 0]]
