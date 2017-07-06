import gc
import os

import numpy as np
from tqdm import tqdm

import medim
from experiments.dl import ModelController
from experiments.config import config_dataset, config_splitter, config_model, \
    config_optimizer, config_batch_iter
from experiments.parser import parse_config


n_epoch = 50
n_iter_per_epoch = 100


def get_dice_threshold(msegms_prob, msegms_true):
    thresholds = []

    n_chans_msegm = len(msegms_prob[0])
    for i in range(n_chans_msegm):
        ps = np.linspace(0, 1, 20)
        best_p = 0
        best_score = 0
        for p in ps:
            score = np.mean(
                [medim.metrics.dice_score(pred[i] > p, true[i])
                 for pred, true in zip(msegms_prob, msegms_true)], axis=0)

            if score > best_score:
                best_p = p
                best_score = score
        print('best dice:', best_score)
        thresholds.append(best_p)
    return np.array(thresholds)


def extract(x, idx):
    return [x[i] for i in idx]

if __name__ == '__main__':
    config = parse_config()
    results_path = config['results_path']

    dataset = config_dataset(config)
    splitter = config_splitter(config)
    optimizer = config_optimizer(config)
    model = config_model(config, optimizer=optimizer,
                         n_chans_in=dataset.n_chans_mscan,
                         n_chans_out=dataset.n_chans_msegm)

    batch_iter = config_batch_iter(config)

    train_val_test = splitter(dataset)

    for i, (train, val, test) in enumerate(train_val_test):
        log_path = os.path.join(results_path, str(i+1))

        mscans_val = [dataset.load_mscan(p) for p in val]
        msegms_val = [dataset.load_msegm(p) for p in val]
        print(msegms_val[0], msegms_val[0].dtype)

        result = []

        with ModelController(model, log_path) as mc:
            for _ in range(n_epoch):
                iterator = batch_iter(train, dataset)

                with iterator:
                    mc.train(iterator, 0.1, n_iter_per_epoch)

                msegms_pred, loss = mc.validate(mscans_val, msegms_val)

            threshold = get_dice_threshold(msegms_pred, msegms_val)
            print('threshold', threshold)

            test_dices = []
            for test_patient_id in test:
                mscan = dataset.load_mscan(test_patient_id)
                msegm = dataset.load_msegm(test_patient_id)

                msegm_pred = mc.predict_object(mscan)
                msegm_pred = msegm_pred > threshold[:, None, None, None]

                dices = medim.metrics.multichannel_dice_score(msegm_pred, msegm)
                print(f'Test dice is {dices} on {test_patient_id}')
                test_dices.append(dices)

            test_dices = np.mean(test_dices, axis=0)
            with open(f'cv_{i+1}', 'w') as f:
                f.write(str(test_dices))

            print('mean dices:', test_dices)

            result.append(test_dices)

        print('Collecting memory:', gc.collect())

        print(result)
        print(np.mean(result, axis=0))
        with open('result', 'w') as f:
            f.write(str(np.mean(result, axis=0)))


