from sklearn.model_selection import KFold, train_test_split

from ..datasets import Dataset


def get_cv_111(dataset: Dataset, *, val_size, n_splits):
    ids = dataset.patient_ids
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=17)

    train_val_test = []
    for train, test in cv.split(ids):
        train = [ids[i] for i in train]
        test = [ids[i] for i in test]
        train, val = train_test_split(train, test_size=val_size,
                                      random_state=25)
        train_val_test.append((list(train), list(val), list(test)))

    return train_val_test
