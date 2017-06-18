"""
Model controller - incapsulate train/val methods for NN model.
"""

import torch
import numpy as np
from tqdm import tqdm

from data_utils import random_nonzero_crops, divide, combine, augment
from nn_utils import iterate_minibatches
from pytorch_utils import stochastic_step, to_numpy, to_var
from mulptiprocessing_utils import par_iterate_minibatches


class Model_controller():
    def __init__(self, model):  # , iterator):
        self.model = model
#         self.iterator = iterator

    def init_train_procedure(self, batch_size=20, lr=0.01,
                             num_of_patches=500, crops_shape=(52, 52, 44),
                             lr_decayer=lambda x: x * 0.5):
        """
        Init params.
        """
        self.initialized = True
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.num_of_patches = num_of_patches
        self.crops_shape = crops_shape
        self.lr_decayer = lr_decayer

    def _train(self, X_train, y_train):
        self.model.train()
        mean_loss, step = 0, 0
        inputs, targets = random_nonzero_crops(X_train, y_train,
                                               self.num_of_patches,
                                               shape=self.crops_shape)
        # par_iterate_minibatches
        bar = tqdm(par_iterate_minibatches(inputs, targets, self.batch_size, augment),
                   total=inputs.shape[0] // self.batch_size)
        for x, y_true in bar:
            batch_loss_val = stochastic_step(x, y_true, self.model, self.optimizer)
            bar.set_description(str(batch_loss_val)[:6])
            mean_loss += batch_loss_val
            step += 1
        return mean_loss / step

    def _validate(self, X_val, y_val):
        self.model.eval()
        mean_loss, step = 0, 0
        inputs, targets = random_nonzero_crops(X_val, y_val,
                                               self.num_of_patches // 4,
                                               shape=self.crops_shape)

        bar = tqdm(iterate_minibatches(inputs, targets, self.batch_size, False),
                   total=inputs.shape[0] // self.batch_size)
        for x, y_true in bar:
            batch_loss_val = stochastic_step(x, y_true, self.model,
                                             self.optimizer, train=False)
            bar.set_description(str(batch_loss_val)[:6])
            mean_loss += batch_loss_val
            step += 1
        return mean_loss / step


    def train(self, X_train, y_train, epoches=50, X_val=None, y_val=None, verbose=True):
        if not self.initialized:
            self.init_train_procedure()
        for epoch in range(epoches):
            tr_loss = self._train(X_train, y_train)
            # TODO hard negative mining!
            if verbose:
                print(str(epoch)+' epoch: \ntrain loss', tr_loss)
            if not X_val is None:
                val_loss = self._validate(X_val, y_val)
                if verbose:
                    print('val loss', val_loss)
            if epoch % 9 == 0:
                new_lr = self.lr_decayer(self.lr)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=new_lr)


    def predict(self, X, zero_padding=[0] * 5, n_parts_per_axis = [1, 1, 4, 4, 2]):
        """
        Partite X into n_parts_per_axis with padding
        """

        a_parts = divide(X, zero_padding, n_parts_per_axis)
        pred = []
        #### PyTorch only
        self.model.eval()
        for i in a_parts:
            i = to_var(i).cuda()
            pred.append(to_numpy(self.model(i)))
            #### PyTorch only
        a_2 = combine(pred, n_parts_per_axis)
        y_pred = np.logical_and(a_2[:, 1, ...] > a_2[:, 0, ...],
                                a_2[:, 1, ...] > a_2[:, 2, ...])
        return y_pred