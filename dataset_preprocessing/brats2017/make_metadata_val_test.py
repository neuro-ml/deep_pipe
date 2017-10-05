import os
import argparse

import pandas as pd

MODALITIES_POSTFIXES = {
    't1': '_t1.nii.gz',
    't1ce': '_t1ce.nii.gz',
    't2': '_t2.nii.gz',
    'flair': '_flair.nii.gz',
}


def make_metadata(patients, age_data) -> pd.DataFrame:
    records = {}
    for patient in patients:
        survival_record = age_data.get(patient, {})

        record = {'age': survival_record.get('Age', None)}
        for modality, postfix in MODALITIES_POSTFIXES.items():
            record[modality] = os.path.join(patient, patient + postfix)

        records[patient] = record

    metadata = pd.DataFrame.from_dict(records, 'index')
    metadata.sort_index(inplace=True)
    return metadata


if __name__ == '__main__':
    # Parsing
    parser = argparse.ArgumentParser('BraTS-2017 val or test preprocessing')
    parser.add_argument('data_path', type=str, help='data_path')

    args = parser.parse_args()
    data_path = args.data_path

    age_data = pd.read_csv(os.path.join(data_path, 'survival_evaluation.csv'),
                           index_col='Brats17ID').to_dict('index')

    metadata = make_metadata(os.listdir(data_path), age_data)
    metadata.to_csv(os.path.join(data_path, 'metadata.csv'), index_label='id')
