from os.path import join

import numpy as np
import pandas as pd


class Brats:
    def __init__(self, processed_path):
        self.processed_path = processed_path
        self.metadata = pd.read_csv(join(processed_path, 'metadata.csv'),
                                    index_col='id')
        self.patients = self.metadata.index.values

    def build_dataname(self, patient):
        return join(self.processed_path, 'data', patient)

    def load_mscan(self, patient):
        dataname = self.build_dataname(patient)
        return np.load(dataname + '_mscan.npy')

    def load_msegm(self, patient):
        dataname = self.build_dataname(patient)
        return np.load(dataname + '_msegm.npy')

    def load_segm(self, patient):
        dataname = self.build_dataname(patient)
        return np.load(dataname + '_segm.npy')
