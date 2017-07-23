import pandas as pd
import numpy as np
import os
import nibabel as nib
from scipy.ndimage import zoom

from dpipe.modules.datasets.factories import FromDataFrame, Scaled
from .base import Dataset


# TODO: subclass those from FromDataFrame class
class Isles(Dataset):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.data_path = data_path
        self.metadata = pd.read_csv(os.path.join(data_path, self.filename))
        self.metadata['id'] = self.metadata.id.astype(str)
        self._patient_ids = self.metadata.index.values

    def load_mscan(self, patient_id):
        id = patient_id
        channels = self.metadata[self.metadata.id == id][self.modalities]
        channels = channels.as_matrix()[0]
        res = []

        for image in channels:
            image = os.path.join(self.data_path, image)
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
        channels = self.metadata[self.metadata.id == patient][self.labels]
        channels = channels.as_matrix()[0]
        res = []

        for image in channels:
            image = os.path.join(self.data_path, image)
            x = nib.load(image).get_data()
            x = self.adjust(x, True)
            # in case adjustment spoils the labels
            x = x > .5
            res.append(x)

        return np.array(res, dtype=bool)

    def segm2msegm(self, segm):
        #  and here too
        pass

    @property
    def patient_ids(self):
        return self.metadata.id.as_matrix()

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


class Isles2017(FromDataFrame, Scaled):
    modality_cols = ['ADC', 'MTT', 'TTP', 'Tmax', 'rCBF', 'rCBV']
    target_cols = ['OT']
    filename = 'meta2017.csv'
    global_path = False
    spacial_shape = 192, 192
    axes = -3, -2


class ISles2017Augmented(Isles2017):
    filename = 'isles2017_augmented.csv'

    @property
    def groups(self):
        return self.dataFrame.patient.as_matrix()


class Isles2017Crop(FromDataFrame):
    modality_cols = ['ADC', 'MTT', 'TTP', 'Tmax', 'rCBF', 'rCBV']
    target_cols = ['OT']
    filename = 'isles2017_crop.csv'
    global_path = False


class ISles2017CropAugmented(Isles2017Crop):
    filename = 'isles2017_crop_augm.csv'

    @property
    def groups(self):
        return self.dataFrame.patient.as_matrix()
