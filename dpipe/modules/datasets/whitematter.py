import os
import numpy as np
import nibabel as nib

from .base import Dataset


class WhiteMatterHyperintensity(Dataset):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.data_path = data_path
        self.idx_to_path = self._get_idx_to_paths(data_path)
        self._patient_ids = list(self.idx_to_path.keys())

    def _get_idx_to_paths(self, path):
        idx_paths = {}
        for path_, dirs, files in os.walk(path):
            if 'orig' in dirs and 'pre' in dirs:
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
            # path_to_brainmask = ???
            res = []
            for modalities in ['pre/FLAIR.nii.gz', 'pre/T1.nii.gz']:
                image = os.path.join(path_to_modalities, modalities)
                x = nib.load(image).get_data().astype('float32')
                x = self._reshape_to(x, new_shape=self.spatial_size)
                # if modalities == 'FLAIR.nii.gz':
                #     mask = nib.load(path_to_brainmask).get_data()
                #     mask = self._reshape_to(mask, new_shape=self.spatial_size)
                #     x[mask==0] = 0
                img_std = x.std()
                x = x / img_std
                res.append(x)
            return np.asarray(res)

    def load_segm(self, patient_id):
        pass

    def load_msegm(self, patient_id):
        path_to_modalities = self.idx_to_path[patient_id]
        res = []
        x = nib.load(os.path.join(path_to_modalities, 'wmh.nii.gz')).get_data()
        x = self._reshape_to(x, new_shape=self.spatial_size)
        res.append(x)
        return np.array(res, dtype=bool)

    def segm2msegm(self, segm):
        pass

    @property
    def patient_ids(self):
        return self._patient_ids

    @property
    def n_chans_mscan(self):
        return 2

    @property
    def n_chans_msegm(self):
        return 1

    @property
    def n_classes(self):
        return 3

    @property
    def spatial_size(self):
        return (256, 256, 84)

