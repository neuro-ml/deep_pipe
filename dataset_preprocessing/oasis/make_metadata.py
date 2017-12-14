import os
import fnmatch
import nibabel as nib
import pandas as pd
import argparse


class PathHelper(object):
    def __init__(self, parent_path, data_dir_name):
        self.parent_path = parent_path
        self.data_dir_name = data_dir_name

    def record_path(self, disc, record):
        return os.path.join(self.data_path(), disc, record)

    def disc_path(self, disc):
        return os.path.join(self.data_path(), disc)

    def data_path(self):
        return os.path.join(self.parent_path, self.data_dir_name)

    def path_for_csv(self, disc, record, file):
        return os.path.join(self.data_dir_name, disc, record, file)


def get_rel_image_path(path, directory, pattern) -> str:
    for file in os.listdir(os.path.join(path, directory)):
        if fnmatch.fnmatch(file, pattern):
            return os.path.join(directory, file)


def process_record(record, disc, path_helper):
    record_path = path_helper.record_path(disc, record)

    s_rel_path = get_rel_image_path(record_path, "PROCESSED/MPRAGE/T88_111", "{}_mpr_n*_111_t88_gfc.hdr".format(record))
    nib.load(os.path.join(record_path, s_rel_path))  # Checking validity
    s_csv_path = path_helper.path_for_csv(disc, record, s_rel_path)

    t_rel_path = get_rel_image_path(record_path, "FSL_SEG", "{}_mpr_n*_anon_111_t88_masked_gfc_fseg.hdr".format(record))
    nib.load(os.path.join(record_path, t_rel_path))   # Checking validity
    t_csv_path = path_helper.path_for_csv(disc, record, t_rel_path)

    return s_csv_path, t_csv_path


def process_disc_dir(disc, path_helper, records) -> None:
    for record in os.listdir(path_helper.disc_path(disc)):
        if record == ".DS_Store":
            continue
        if fnmatch.fnmatch(record, "OAS*"):
            s_path, t_path = process_record(record, disc, path_helper)
            records[record] = {'S': s_path, 'T': t_path}
        else:
            raise RuntimeError


def create_metadata_csv(parent_dir_path, first_disc, disc_num, data_dir_name="data"):
    records = {}
    path_helper = PathHelper(parent_dir_path, data_dir_name)

    for disc_num in range(first_disc, first_disc + disc_num):
        disc = "disc{}".format(disc_num)
        print("processing {}".format(disc))
        process_disc_dir(disc, path_helper, records)

    metadata = pd.DataFrame.from_dict(records, 'index')
    metadata.sort_index(inplace=True)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to a dir with "data" folder')
    parser.add_argument('--first_disc', type=int, default=1)
    parser.add_argument('--disc_num', type=int, default=1)

    args = parser.parse_args()
    parent_dir = args.data_path
    first_disc = args.first_disc
    disc_num = args.disc_num

    assert(1 <= first_disc <= 12 and 1 <= disc_num and first_disc + disc_num - 1 <= 12)
    metadata = create_metadata_csv(parent_dir, first_disc, disc_num)
    metadata.to_csv(os.path.join(parent_dir, 'metadata.csv'), index_label='id')
