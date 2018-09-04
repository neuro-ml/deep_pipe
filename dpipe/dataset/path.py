import os

from dpipe.medim import load_image


def remove_extension(name):
    suffix = True
    while suffix:
        name, suffix = os.path.splitext(name)
    return name


class Folder:
    """Dataset that loads objects from a given folder. The ids are assigned based on filenames."""

    def __init__(self, folder: str):
        self.folder = folder
        files = os.listdir(folder)
        self.ids = tuple(map(remove_extension, files))
        assert len(set(self.ids)) == len(files), f'There are duplicate ids in the folder {folder}'
        self._id_to_file = dict(zip(self.ids, files))

    def load(self, identifier: str, loader=load_image):
        return loader(os.path.join(self.folder, self._id_to_file[identifier]))
