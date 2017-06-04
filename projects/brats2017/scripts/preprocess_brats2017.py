import os
from os.path import join
from collections import namedtuple
import argparse

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

import medim

MODALITIES_POSTFIXES = ['_t1.nii.gz', '_t1ce.nii.gz',
                        '_t2.nii.gz', '_flair.nii.gz']
SEGMENTATION_POSTFIX = '_seg.nii.gz'


def load_modality(patient, patient_path, postfix):
    filepath = join(patient_path, patient + postfix)
    data = nib.load(filepath).get_data()
    return data


class RawDataLoader:
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

        self.survival_data = pd.read_csv(raw_data_path+'/survival_data.csv',
                                         index_col='Brats17ID')

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


Patient = namedtuple('Patient', ['id', 'type', 'age', 'survival'])


def make_metadata(patients, hgg_patients, survival_data):
    print('Building metadata')
    records= []
    for patient_name in patients:
        type = 'hgg' if patient_name in hgg_patients else 'lgg'
        if patient_name in survival_data.index:
            age = survival_data.loc[patient_name].Age
            survival = survival_data.loc[patient_name].Survival
        else:
            age = None
            survival = None

        records.append(Patient(patient_name, type, age, survival))

    metadata = pd.DataFrame(records)
    metadata.index = metadata.id
    metadata = metadata.drop(['id'], axis=1)
    return metadata


def encode_msegm(s):
    r = np.zeros((3, *s.shape), dtype=bool)
    r[0] = s > 0
    r[1] = (s == 1) | (s == 4)
    r[2] = (s == 4)
    return r


def preprocess(data_loader, processed_path):
    patients = data_loader.patients
    print('Preprocessing')
    for patient in tqdm(patients):
        mscan = data_loader.load_mscan(patient)
        segmentation = data_loader.load_segmentation(patient)

        # Extract part with the brain
        mask = np.any(mscan, axis=0)
        mscan, segmentation = medim.bb.extract(mscan, segmentation, mask=mask)

        mscan = medim.prep.normalize_mscan(mscan, mean=False)
        msegm = encode_msegm(segmentation)

        filename = join(processed_path, 'data', patient)
        np.save(filename+'_mscan', mscan)
        np.save(filename+'_msegm', msegm)


if __name__ == '__main__':
    # Parsing
    parser = argparse.ArgumentParser('BraTS-2017 preprocess')
    parser.add_argument('raw_path', type=str, help='raw data path')
    parser.add_argument('processed_path', type=str, help='processed data path')

    args = parser.parse_args()
    raw_data_path = args.raw_path
    processed_path = args.processed_path

    data_loader = RawDataLoader(raw_data_path)

    metadata = make_metadata(data_loader.patients, data_loader.hgg_patients,
                       data_loader.survival_data)
    metadata.to_csv(join(processed_path, 'metadata.csv'))

    preprocess(data_loader, processed_path)
