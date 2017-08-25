import pandas as pd
import numpy as np
import os
import nibabel as nib

from .base import Dataset


class FromDataFrame(Dataset):
    filename = None
    target_cols = None
    modality_cols = None
    group_col = None
    global_path = False

    def __init__(self, data_path, modalities=None, targets=None):
        self.data_path = data_path
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

    def load_by_paths(self, paths):
        res = []
        for image in paths:
            if not self.global_path:
                image = os.path.join(self.data_path, image)
            x = self.load_channel(image)
            res.append(x)

        return np.asarray(res)

    def load_msegm(self, patient):
        # super method is needed here to allow chaining in its children
        super().load_msegm(patient)

        paths = self.dataFrame[self.target_cols].loc[patient]
        image = self.load_by_paths(paths)

        return image >= .5

    def load_mscan(self, patient_id):
        super().load_mscan(patient_id)

        paths = self.dataFrame[self.modality_cols].loc[patient_id]
        image = self.load_by_paths(paths)
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

    @staticmethod
    def build(_filename, _target_cols, _modality_cols, _global_path=False):
        class Child(FromDataFrame):
            filename = _filename
            target_cols = _target_cols
            modality_cols = _modality_cols
            global_path = _global_path

        return Child


def set_filename(dataset, _filename):
    class Wrapped(dataset):
        filename = _filename

    return Wrapped


# decorator for decorators. yeah, baby
def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


@parametrized
def apply(cls, func, **kwargs):
    class Wrapped(cls):
        def load_mscan(self, patient_id):
            image = super().load_mscan(patient_id)
            return func(image, **kwargs)

        def load_msegm(self, patient):
            image = super().load_msegm(patient)
            image = func(image, **kwargs)

            return image >= .5

    return Wrapped


@parametrized
def mscan(cls, func, **kwargs):
    class Wrapped(cls):
        def load_mscan(self, patient_id):
            image = super().load_mscan(patient_id)
            return func(image, **kwargs)

    return Wrapped


@parametrized
def msegm(cls, func, **kwargs):
    class Wrapped(cls):
        def load_msegm(self, patient):
            image = super().load_msegm(patient)
            image = func(image, **kwargs)

            return image >= .5

    return Wrapped


def append_channels(cls):
    class Wrapped(cls):
        def __init__(self, *args, append_paths, **kwargs):
            super(Wrapped, self).__init__(*args, **kwargs)
            self.append_paths = append_paths

        def load_mscan(self, patient_id):
            image = super().load_mscan(patient_id)

            additional = [i % patient_id for i in self.append_paths]
            second = self.load_by_paths(additional)
            if second.ndim != image.ndim:
                second = second[0]
            image = np.vstack((image, second))
            image = image.astype('float32')

            return image

        @property
        def n_chans_mscan(self):
            return len(self.append_paths) + super().n_chans_mscan

    return Wrapped
