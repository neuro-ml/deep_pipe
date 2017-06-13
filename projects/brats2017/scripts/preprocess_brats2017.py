import os
from os.path import join
from collections import namedtuple
import argparse

import numpy as np
import pandas as pd
import nibabel as nib

from utils import preprocess, RawDataLoader


MODALITIES_POSTFIXES = ['_t1.nii.gz', '_t1ce.nii.gz',
                        '_t2.nii.gz', '_flair.nii.gz']
SEGMENTATION_POSTFIX = '_seg.nii.gz'

SCAN_SIZE = [157, 189, 149]

def load_modality(patient, patient_path, postfix):
    filepath = join(patient_path, patient + postfix)
    data = nib.load(filepath).get_data()
    return data


def load_mscan(patient, patient_path):
    scans = [load_modality(patient, patient_path, p)
             for p in MODALITIES_POSTFIXES]
    return np.array(scans, dtype=np.float32)


def load_segm(patient, patient_path):
    segm = load_modality(patient, patient_path, SEGMENTATION_POSTFIX)
    segm = np.array(segm, dtype=np.uint8)
    segm[segm == 4] = 3
    return segm


def get_survival_class(survival):
    if np.isnan(survival):
        return None
    if survival > 18 * 30:
        return 2
    elif survival < 6 * 30:
        return 0
    else:
        return 1


Patient = namedtuple('Patient', ['id', 'type', 'age', 'survival'])
def make_metadata(patients, hgg_patients, survival_data) -> pd.DataFrame:
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

    metadata['survival_class'] = metadata.survival.apply(get_survival_class)

    return metadata


if __name__ == '__main__':
    # Parsing
    parser = argparse.ArgumentParser('BraTS-2017 preprocess')
    parser.add_argument('raw_path', type=str, help='raw data path')
    parser.add_argument('processed_path', type=str, help='processed data path')

    args = parser.parse_args()
    raw_data_path = args.raw_path
    processed_path = args.processed_path

    data_loader = RawDataLoader(raw_data_path, load_mscan, load_segm)

    survival_data = pd.read_csv(join(raw_data_path, 'survival_data.csv'),
                                     index_col='Brats17ID')
    metadata = make_metadata(data_loader.patients, data_loader.hgg_patients,
                             survival_data)
    metadata.to_csv(join(processed_path, 'metadata.csv'))

    preprocess(data_loader, processed_path, SCAN_SIZE)
