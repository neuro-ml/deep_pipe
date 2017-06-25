from sklearn.model_selection import KFold, train_test_split


def make_cv_111(*, val_size, n_splits):
    def cv_111(dataset):
        total_size = len(dataset.patient_ids)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=17)

        train_val_test = []
        for train, test in cv.split(range(total_size)):
            train, val = train_test_split(train, test_size=val_size,
                                          random_state=25)
            train_val_test.append((list(train), list(val), list(test)))

        return train_val_test
    return cv_111
