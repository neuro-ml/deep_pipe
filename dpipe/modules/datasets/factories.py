import pandas as pd
import numpy as np
import os
import nibabel as nib
from scipy import ndimage

from .base import Dataset


# refactor the fuck out of this
class FromDataFrame(Dataset):
    filename = None
    target_cols = None
    modality_cols = None
    group_col = None
    global_path = False

    def __init__(self, data_path, modalities=None, targets=None):
        super().__init__(data_path)
        df = pd.read_csv(os.path.join(data_path, self.filename))
        df['id'] = df.id.astype(str)
        self._patient_ids = df.id.as_matrix()
        self.dataFrame = df.set_index('id')
        if modalities is not None:
            self.modality_cols = modalities
        if targets is not None:
            self.target_cols = targets

    @staticmethod
    def load_channel(path):
        if path.endswith('.npy'):
            return np.load(path)
        return nib.load(path).get_data()

    def load_by_cols(self, patient_id, columns):
        channels = self.dataFrame[columns].loc[patient_id]
        res = []

        for image in channels:
            if not self.global_path:
                image = os.path.join(self.data_path, image)
            x = self.load_channel(image)
            res.append(x)

        return np.asarray(res)

    def load_msegm(self, patient):
        # super method is needed here to allow chaining in its children
        super().load_msegm(patient)

        image = self.load_by_cols(patient, self.target_cols)

        return image >= .5

    def load_mscan(self, patient_id):
        super().load_mscan(patient_id)

        image = self.load_by_cols(patient_id, self.modality_cols)
        image = image.astype('float32')

        axes = tuple(range(1, image.ndim))
        m = image.min(axis=axes, keepdims=True)
        M = image.max(axis=axes, keepdims=True)

        return (image - m) / (M - m)

    def load_segm(self, patient_id):
        super().load_segm(patient_id)

    def segm2msegm(self, segm):
        super().segm2msegm(segm)

    @property
    def patient_ids(self):
        return self._patient_ids

    @property
    def n_chans_mscan(self):
        return len(self.modality_cols)

    @property
    def n_chans_msegm(self):
        return len(self.target_cols)

    @property
    def n_classes(self):
        return len(self.target_cols)

    @property
    def groups(self):
        """
        Used for GroupKFold
        """
        assert self.group_col is not None
        return self.dataFrame[self.group_col].as_matrix()

        # @staticmethod
        # def build(_filename, _target_cols, _modality_cols, _global_path=False):
        #     class Child(FromDataFrame):
        #         filename = _filename
        #         target_cols = _target_cols
        #         modality_cols = _modality_cols
        #         global_path = _global_path
        #
        #     return Child


# TODO: unify those two classes
class Scaled(FromDataFrame):
    spacial_shape = None
    axes = None

    def __init__(self, data_path, **kwargs):
        super().__init__(data_path, **kwargs)
        if self.axes is not None:
            self.axes = list(sorted(self.axes))

    def scale(self, image, order):
        if self.axes is None:
            l, L = len(self.spacial_shape), image.ndim
            self.axes = list(range(L - l, L))

        old_shape = np.array(image.shape)[self.axes]
        new_shape = np.array(self.spacial_shape)

        scale = np.ones_like(image)
        scale[self.axes] = new_shape / old_shape
        return ndimage.zoom(image, scale, order=order)

    def load_mscan(self, patient_id):
        image = super().load_mscan(patient_id)

        return self.scale(image, 3)

    def load_msegm(self, patient):
        image = super().load_msegm(patient)
        image = self.scale(image, 0)

        return image >= .5

        # @staticmethod
        # def build(_spacial_shape):
        #     class Child(Scaled):
        #         spacial_shape = _spacial_shape
        #
        #     return Child


class Padded(FromDataFrame):
    spacial_shape = None
    axes = None

    def __init__(self, data_path, **kwargs):
        super().__init__(data_path, **kwargs)
        if self.axes is not None:
            self.axes = list(sorted(self.axes))

    def pad(self, image):
        if self.axes is None:
            l, L = len(self.spacial_shape), image.ndim
            self.axes = list(range(L - l, L))

        old_shape = np.array(image.shape)[self.axes]
        new_shape = np.array(self.spacial_shape)

        assert (old_shape <= new_shape).all()

        delta = new_shape - old_shape
        pad = np.array((delta // 2, (delta + 1) // 2)).T

        padding = np.zeros((image.ndim, 2), int)
        padding[self.axes] = pad.astype(int)

        return np.pad(image, padding, mode='constant')

    def load_mscan(self, patient_id):
        image = super().load_mscan(patient_id)

        return self.pad(image)

    def load_msegm(self, patient):
        image = super().load_msegm(patient)
        image = self.pad(image)

        return image >= .5

        # @staticmethod
        # def build(_spacial_shape):
        #     class Child(Scaled):
        #         spacial_shape = _spacial_shape
        #
        #     return Child
