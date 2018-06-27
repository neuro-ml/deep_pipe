import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, GroupShuffleSplit


class ShuffleGroupKFold(KFold):
    def split(self, *, X, groups):
        names_unique = np.unique(groups)
        for train, test in super().split(names_unique):
            train = np.in1d(groups, names_unique[train])
            test = np.in1d(groups, names_unique[test])
            yield np.where(train)[0], np.where(test)[0]


def train_test_split_groups(X, *, val_size, groups=None, **kwargs):
    split_class = (ShuffleSplit if groups is None else GroupShuffleSplit)
    split = split_class(test_size=val_size, **kwargs)
    train, val = next(split.split(X=X, groups=groups))
    return X[train], X[val]
