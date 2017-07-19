import os
import pandas as pd
import numpy as np
from os.path import join
import nibabel as nib
from scipy.ndimage import zoom

from .base import Dataset


class WhiteMatterHyperintensity(Dataset):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.data_path = data_path
        self.metadata = pd.read_csv(data_path)
        self.metadata['idx'] = self.metadata.idx.astype(str)
        self._patient_ids = self.metadata.index.values
    
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
            idx = patient_id
            path_to_modalities = self.metadata[self.metadata.idx == idx].path_imgs.as_matrix()[0]
            path_to_brainmask = self.metadata[self.metadata.idx == idx].path_brain_mask.as_matrix()[0]
            res = []

            hardcoded_shape = self.spatial_size #(256, 256, 84)

            for modalities in ['FLAIR.nii.gz', 'T1.nii.gz']:
                image = os.path.join(path_to_modalities, modalities)
                x = nib.load(image).get_data().astype('float32')
                x = self._reshape_to(x, new_shape=hardcoded_shape)

                if modalities == 'FLAIR.nii.gz':
                    mask = nib.load(path_to_brainmask).get_data()
                    mask = self._reshape_to(mask, new_shape=hardcoded_shape)
                    x[mask==0] = 0

                img_std = x.std()
                x = x / img_std
                res.append(x)

            return np.asarray(res)

    def load_segm(self, patient_id):
        # dunno what to do here
        pass

    def load_msegm(self, patient):
        path = self.metadata[self.metadata.idx == patient].path_segm_mask.as_matrix()[0]
        res = []
        
        x = nib.load(path).get_data()
        x = self._reshape_to(x, new_shape=self.spatial_size)
        res.append(x)

        return np.array(res, dtype=bool)

    def segm2msegm(self, segm):
        #  and here too
        pass

    @property
    def patient_ids(self):
        return self.metadata.idx.as_matrix()

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

