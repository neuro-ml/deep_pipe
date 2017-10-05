import os
import argparse

import pandas as pd

MODALITIES_POSTFIXES = {
    't1': '_t1.nii.gz',
    't1ce': '_t1ce.nii.gz',
    't2': '_t2.nii.gz',
    'flair': '_flair.nii.gz',
    'segm': '_seg.nii.gz'
}

SEGMENTATION_POSTFIX = '_seg.nii.gz'


def get_survival_class(survival):
    if survival is None:
        return None
    if survival > 18 * 30:
        return 2
    elif survival < 6 * 30:
        return 0
    else:
        return 1


def make_metadata(hgg_patients, lgg_patients, survival_data) -> pd.DataFrame:
    records = {}
    for patient in hgg_patients + lgg_patients:
        survival_record = survival_data.get(patient, {})

        record = {'cancer_type': 'HGG' if patient in hgg_patients else 'LGG'}
        record['age'] = survival_record.get('Age', None)
        record['survival_days'] = survival_record.get('Survival', None)
        record['survival_class'] = get_survival_class(record['survival_days'])
        for modality, postfix in MODALITIES_POSTFIXES.items():
            record[modality] = os.path.join(record['cancer_type'], patient,
                                            patient + postfix)

        records[patient] = record

    metadata = pd.DataFrame.from_dict(records, 'index')
    metadata.sort_index(inplace=True)
    return metadata


if __name__ == '__main__':
    # Parsing
    parser = argparse.ArgumentParser('BraTS-2017 training data preprocessing')
    parser.add_argument('data_path', type=str, help='data_path')

    args = parser.parse_args()
    data_path = args.data_path

    hgg_patients = os.listdir(os.path.join(data_path, 'HGG'))
    lgg_patients = os.listdir(os.path.join(data_path, 'LGG'))

    survival_data = pd.read_csv(
        os.path.join(data_path, 'survival_data.csv'), index_col='Brats17ID'
    ).to_dict('index')

    metadata = make_metadata(hgg_patients, lgg_patients, survival_data)
    metadata.to_csv(os.path.join(data_path, 'metadata.csv'), index_label='id')
