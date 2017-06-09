import os
from os.path import join

import numpy as np
from tqdm import tqdm

import medim


class RawDataLoader:
    def __init__(self, raw_data_path, load_mscan, load_segm):
        self.raw_data_path = raw_data_path
        self._load_mscan = load_mscan
        self._load_segm = load_segm

        hgg_path = join(raw_data_path, 'HGG')
        lgg_path = join(raw_data_path, 'LGG')

        self.hgg_patients = os.listdir(hgg_path)
        self.lgg_patients = os.listdir(lgg_path)
        self.patients = self.hgg_patients + self.lgg_patients

        self.patient2path = {patient: join(hgg_path, patient)
                             for patient in self.hgg_patients}
        self.patient2path.update({patient: join(lgg_path, patient)
                                  for patient in self.lgg_patients})

    def load_mscan(self, patient):
        patient_path = self.patient2path[patient]
        return self._load_mscan(patient, patient_path)


    def load_segm(self, patient):
        patient_path = self.patient2path[patient]
        return self._load_segm(patient, patient_path)


def preprocess(raw_data_loader: RawDataLoader, processed_path, scan_size):
    patients = raw_data_loader.patients
    print('Preprocessing')
    for patient in tqdm(patients):
        mscan = raw_data_loader.load_mscan(patient)
        segm = raw_data_loader.load_segm(patient)

        # Extract part with the brain
        mask = np.any(mscan, axis=0)
        mscan, segm = medim.bb.extract_fixed(mscan, segm, mask=mask,
                                             size=scan_size)

        mscan = medim.prep.normalize_mscan(mscan, mean=False)

        filename = join(processed_path, 'data', patient)
        np.save(filename + '_mscan', mscan)
        np.save(filename + '_segm', segm)
