import numpy as np
from sklearn.model_selection import KFold, train_test_split

from dpipe.dataset import Dataset


def get_los_cv(dataset: Dataset):
    ids = dataset.patient_ids
    Sing = ['61', '53', '60', '68', '54', '59', '62', '65', '55', '63', '56', '66', '58', '57', '69', '67', '51', '50', '52', '64']
    Amst = ['114', '100', '107', '102', '101', '112', '113', '108', '116', '126', '109', '106', '104', '110', '137', '144', '103', '115', '132', '105']
    Utr = ['41', '2', '49', '19', '39', '31', '35', '21', '6', '29', '25', '0', '17', '11', '27', '23', '4', '37', '8', '33']
    
    train_val_test = []
    train = [i for i in Amst+Utr]
    test = [i for i in Sing]
    val = [train[0], train[-1], train[1], train[-2]]
    [train.remove(i) for i in val]
    train_val_test.append((list(train), list(val), list(test)))
    train = [i for i in Sing+Utr]
    test = [i for i in Sing]
    val = [train[0], train[-1], train[1], train[-2]]
    [train.remove(i) for i in val]
    train_val_test.append((list(train), list(val), list(test)))
    train = [i for i in Amst+Sing]
    test = [i for i in Sing]
    val = [train[0], train[-1], train[1], train[-2]]
    [train.remove(i) for i in val]
    train_val_test.append((list(train), list(val), list(test)))
    
    return train_val_test
