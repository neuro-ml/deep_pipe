from sklearn.model_selection import KFold

from dpipe.dataset import Dataset


def get_cv_11(dataset: Dataset, *, n_splits):
    ids = dataset.patient_ids
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=17)

    return list(cv.split(ids))
