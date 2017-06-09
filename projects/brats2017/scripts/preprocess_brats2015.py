import os
from os.path import join
from collections import namedtuple
import argparse

import numpy as np
import pandas as pd
import SimpleITK

from utils import preprocess, RawDataLoader


MODALITIES_POSTFIXES = ['XX.O.MR_T1.', 'XX.O.MR_T1c.',
                        'XX.O.MR_T2.', 'XX.O.MR_Flair.']
SEGMENTATION_POSTFIX = '.OT.'

SCAN_SIZE = [146, 181, 160]


def load_modality(patient, patient_path, postfix):
    dirs = os.listdir(patient_path)
    dirs = [d for d in dirs if postfix in d]
    if len(dirs) != 1:
        print(patient, dirs)
        assert len(dirs) == 1
    filename = join(patient_path, dirs[0], dirs[0]+'.mha')

    scan = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(filename))
    return scan


def load_mscan(patient, patient_path):
    scans = [load_modality(patient, patient_path, p)
             for p in MODALITIES_POSTFIXES]
    return np.array(scans, dtype=np.float32)


def load_segm(patient, patient_path):
    segm = load_modality(patient, patient_path, SEGMENTATION_POSTFIX)
    return np.array(segm, dtype=np.uint8)


Patient = namedtuple('Patient', ['id', 'type'])
def make_metadata(patients, hgg_patients) -> pd.DataFrame:
    print('Building metadata')
    records= []
    for patient_name in patients:
        type = 'hgg' if patient_name in hgg_patients else 'lgg'

        records.append(Patient(patient_name, type))

    metadata = pd.DataFrame(records)
    metadata.index = metadata.id
    metadata = metadata.drop(['id'], axis=1)
    return metadata


if __name__ == '__main__':
    # Parsing
    parser = argparse.ArgumentParser('BraTS-2015 preprocess')
    parser.add_argument('raw_path', type=str, help='raw data path')
    parser.add_argument('processed_path', type=str, help='processed data path')

    args = parser.parse_args()
    raw_data_path = args.raw_path
    processed_path = args.processed_path

    data_loader = RawDataLoader(raw_data_path, load_mscan, load_segm)

    metadata = make_metadata(data_loader.patients, data_loader.hgg_patients)
    metadata.to_csv(join(processed_path, 'metadata.csv'))

    preprocess(data_loader, processed_path, SCAN_SIZE)
