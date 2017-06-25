import pandas as pd
import numpy as np
from os.path import join
import nibabel as nib
from scipy.ndimage import zoom

from .base import Dataset


class Isles(Dataset):
    def __init__(self, processed_path):
        super().__init__(processed_path)
        self.processed_path = processed_path
        self.metadata = pd.read_csv(join(processed_path, self.filename))
        self._patient_ids = self.metadata.index.values

    def load_mscan(self, patient_id):
        channels = self.metadata.iloc[patient_id][self.modalities]
        res = []

        for image in channels:
            image = image.replace('data/', self.processed_path)
            x = nib.load(image).get_data()
            x = self.adjust(x)
            x = x.astype('float32')
            m, M = x.min(), x.max()
            x = (x - m) / (M - m)
            res.append(x)

        return np.asarray(res)

    def load_segm(self, patient_id):
        # dunno what to do here
        pass

    def load_msegm(self, patient):
        channels = self.metadata.iloc[patient][self.labels]
        res = []

        for image in channels:
            image = image.replace('data/', self.processed_path)
            x = nib.load(image).get_data()
            x = self.adjust(x, True)
            res.append(x)

        return np.asarray(res)

    def segm2msegm(self, segm):
        #  and here too
        pass

    @property
    def patient_ids(self):
        return list(range(len(self.metadata)))

    @property
    def n_chans_mscan(self):
        return len(self.modalities)

    @property
    def n_chans_msegm(self):
        return len(self.labels)

    @property
    def n_classes(self):
        return len(self.labels)


def siss_factory(file):
    class IslesSISS(Isles):
        modalities = ['T1', 'T2', 'Flair', 'DWI']
        labels = ['OT']
        filename = file

        def adjust(self, x, label=False):
            if x.shape[-1] != 154:
                x = np.pad(x, ((0, 0), (0, 0), (0, 1)), mode='constant')
            return x

        @property
        def spatial_size(self):
            return 230, 230, 154

    return IslesSISS


def spes_factory(file):
    class IslesSPES(Isles):
        modalities = ['CBF', 'CBV', 'DWI', 'T1c', 'T2', 'TTP', 'Tmax']
        labels = ['penumbralabel', 'corelabel']
        filename = file

        def adjust(self, x, label=False):
            ref_shape = np.array(self.spatial_size)
            order = 0 if label else 3
            x = zoom(x, ref_shape / x.shape, order=order)
            return x

        @property
        def spatial_size(self):
            return 96, 110, 72

    return IslesSPES
