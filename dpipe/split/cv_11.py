from sklearn.model_selection import KFold

from dpipe.dataset import DataSet


def get_cv_11(dataset: DataSet, *, n_splits):
    ids = dataset.ids
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=17)

    return list(cv.split(ids))
