import numpy as np

from dpipe.medim.metrics import dice_score


def extract(l, idx):
    return [l[i] for i in idx]


def build_slices(start, end):
    assert len(start) == len(end)
    return list(map(slice, start, end))


def calc_max_dices(y_true, y_pred):
    dices = []
    thresholds = np.linspace(0, 1, 20)
    for true, pred in zip(y_true, y_pred):
        temp = []
        for i in range(len(true)):
            temp.append([dice_score(pred[i] > thr, true[i].astype(bool))
                         for thr in thresholds])
        dices.append(temp)
    dices = np.asarray(dices)
    return dices.mean(axis=0).max(axis=1)


def load_image(path):
    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.nii') or path.endswith('.nii.gz'):
        import nibabel as nib
        return nib.load(path).get_data()
    else:
        raise ValueError(f"Couldn't read scan from path: {path}.\n"
                         "Unknown data extension.")
