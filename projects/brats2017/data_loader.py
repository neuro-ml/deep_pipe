import os
from os.path import join

import numpy as np
import pandas as pd
import nibabel as nib


MODALITIES_POSTFIXES = ['_t1.nii.gz', '_t1ce.nii.gz',
                        '_t2.nii.gz', '_flair.nii.gz']
SEGMENTATION_POSTFIX = '_seg.nii.gz'


def load_modality(patient, patient_path, postfix):
    filepath = join(patient_path, patient + postfix)
    data = nib.load(filepath).get_data()
    return data


class DataLoader:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        hgg_path = join(raw_data_path, 'HGG')
        lgg_path = join(raw_data_path, 'LGG')

        self.hgg_patients = os.listdir(hgg_path)
        self.lgg_patients = os.listdir(lgg_path)
        self.patients = self.hgg_patients + self.lgg_patients

        self.patient2path = {patient: join(hgg_path, patient)
                             for patient in self.hgg_patients}
        self.patient2path.update({patient: join(lgg_path, patient)
                                  for patient in self.lgg_patients})

        self.survival_data = pd.read_csv(raw_data_path+'/survival_data.csv')

    def load_mscan(self, patient):
        patient_path = self.patient2path[patient]
        mscan = [load_modality(patient, patient_path, postfix)
                 for postfix in MODALITIES_POSTFIXES]
        mscan = np.array(mscan)
        return mscan

    def load_segmentation(self, patient):
        patient_path = self.patient2path[patient]
        segmentation = np.array(
            load_modality(patient, patient_path, SEGMENTATION_POSTFIX),
            dtype=np.uint8)
        return segmentation


