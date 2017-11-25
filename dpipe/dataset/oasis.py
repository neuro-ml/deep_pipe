import numpy as np
from .from_csv import FromCSVInt
from dpipe.config import register


@register('oasis')
class Oasis(FromCSVInt):
    def __init__(self, data_path, metadata_rpath='metadata.csv'):
        super().__init__(
            data_path=data_path,
            metadata_rpath=metadata_rpath,
            modalities=['S'],
            target='T',
            segm2msegm_matrix=np.diag([0, 1, 1, 1])
        )

    def load_segm(self, patient_id):
        img = super().load_segm(patient_id).astype(int)
        # The loaded image has a shape of (176, 208, 176, 1). Next line remove the last useless dimension.
        img.resize(img.shape[:-1])
        return img

    def load_image(self, patient_id):
        mscan = super().load_image(patient_id)
        assert(len(mscan) == 1)
        scan = mscan[0]
        scan.resize(scan.shape[:-1])
        # Though resizing happens in place, for some reason resizing mscan[0], it's shape claims that it is unchanged.
        # That's why I'm creating a scan variable and returning a new ndarray.
        # returning new array.
        return np.asarray([scan])
