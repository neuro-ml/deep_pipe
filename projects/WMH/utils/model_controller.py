"""
Model controller - incapsulate train/val methods for NN model.
"""
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from data_utils import random_nonzero_crops, random_slicer, divide, combine, augment
from nn_utils import iterate_minibatches
from pytorch_utils import stochastic_step, to_numpy, to_var
from mulptiprocessing_utils import par_iterate_minibatches


class Model_controller():
    def __init__(self, model):  # , iterator):
        self.model = model
#         self.iterator = iterator

    def init_train_procedure(self, batch_size=20, lr=0.01, net='3D',
                             num_of_patches=500, crops_shape=(52, 52, 44),
                             lr_decayer=lambda x: x * 0.95):
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
        self.res = dict()
        assert net=='2D' or net=='3D'
        self.net_dim = net
        self.neg_minig_storage = []


    def  _add_statistics(self, **kwargs):
        for name in kwargs:
            stat_vars = self.res.keys()
            if name in stat_vars:
                self.res[name].append(kwargs[name])
            else:
                self.res[name] = []


    # def _train(self, X_train, y_train):
    #     self.model.train()
    #     mean_loss, step = 0, 0
    #     if self.net_dim == '3D':
    #         inputs, targets = random_nonzero_crops(X_train, y_train,
    #                                                self.num_of_patches,
    #                                                shape=self.crops_shape)
    #     else:
    #         inputs, targets = random_slicer(X_train, y_train)
    #
    #     # bar = tqdm(par_iterate_minibatches(inputs, targets, self.batch_size,
    #     #                                    augment),
    #     #            total=inputs.shape[0] // self.batch_size)
    #     bar = tqdm(iterate_minibatches(inputs, targets, self.batch_size),
    #                total=inputs.shape[0] // self.batch_size)
    #     for x, y_true in bar:
    #         batch_loss_val = stochastic_step(x, y_true, self.model, self.optimizer)
    #         bar.set_description(str(batch_loss_val)[:6])
    #         mean_loss += batch_loss_val
    #         step += 1
    #     return mean_loss / step
    #
    #
    # def _validate(self, X_val, y_val):
    #     self.model.eval()
    #     mean_loss, step = 0, 0
    #     if self.net_dim == '3D':
    #         inputs, targets = random_nonzero_crops(X_val, y_val,
    #                                                self.num_of_patches // 4,
    #                                                shape=self.crops_shape)
    #     else:
    #         inputs, targets = random_slicer(X_val, y_val)
    #
    #     bar = tqdm(iterate_minibatches(inputs, targets, self.batch_size, False),
    #                total=inputs.shape[0] // self.batch_size)
    #     for x, y_true in bar:
    #         batch_loss_val = stochastic_step(x, y_true, self.model,
    #                                          self.optimizer, train=False)
    #         bar.set_description(str(batch_loss_val)[:6])
    #         mean_loss += batch_loss_val
    #         step += 1
    #     return mean_loss / step
    #
    #
    # def train(self, X_train, y_train, epoches=50, X_val=None, y_val=None,
    #           verbose=True):
    #     if not self.initialized:
    #         self.init_train_procedure()
    #
    #     for epoch in range(epoches):
    #         tr_loss = self._train(X_train, y_train)
    #         self._add_statistics(**{'train_loss': tr_loss})
    #         # TODO hard negative mining!
    #         if verbose:
    #             print(str(epoch)+' epoch: \ntrain loss', tr_loss)
    #
    #         if not X_val is None:
    #             val_loss = self._validate(X_val, y_val)
    #             self._add_statistics(**{'val_loss': val_loss})
    #             if verbose:
    #                 print('val loss', val_loss)
    #
    #         if epoch % 5 == 0:
    #             self.lr = self.lr_decayer(self.lr)
    #             self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.lr)
    #         self._add_statistics(**{'lr': self.lr})


    def predict(self, X, zero_padding=[0] * 5, n_parts_per_axis = [1, 1, 4, 4, 2]):
        """
        Divide X into n_parts_per_axis with padding.
        """
        shape = X.shape
        if self.net_dim == '3D':
            a_parts = divide(X, zero_padding, n_parts_per_axis)
        else:
            a_parts = X.reshape(shape[0] * shape[-1], shape[1], shape[2], shape[3])
        pred = []
        #### TODO not PyTorch only
        self.model.eval()
        for i in a_parts:
            i = to_var(i).cuda()
            pred.append(to_numpy(self.model(i)))
        #### PyTorch only
        if self.net_dim == '3D':
            a_2 = combine(pred, n_parts_per_axis)
        else:
            a_2 = np.array(pred).reshape(*shape)
        y_pred = np.logical_and(a_2[:, 1, ...] > a_2[:, 0, ...],
                                a_2[:, 1, ...] > a_2[:, 2, ...])
        return y_pred


#region Hard_Neg_minig!
    #
    def _train(self, X_train, y_train, hard_neg_mining = False):
        self.model.train()
        mean_loss, step = 0, 0
        inputs, targets = random_nonzero_crops(X_train, y_train,
                                               self.num_of_patches,
                                               shape=self.crops_shape)
        if hard_neg_mining and len(self.neg_minig_storage) > 0:
            prev_inp = self.neg_minig_storage[0]
            prev_targ = self.neg_minig_storage[1]
            print('added', len(prev_inp), 'examples')
            inputs = np.concatenate([inputs, prev_inp])
            targets = np.concatenate([targets, prev_targ])

        bar = tqdm(par_iterate_minibatches(inputs, targets, self.batch_size,
                                           augment),
                   total=inputs.shape[0] // self.batch_size)
        for x, y_true in bar:
            batch_loss_val = stochastic_step(x, y_true, self.model, self.optimizer)
            bar.set_description(str(batch_loss_val)[:6])
            mean_loss += batch_loss_val
            step += 1
        if hard_neg_mining:
            _ = self._validate(inputs, targets, hard_neg_mining=True)
        return mean_loss / step


    def _validate(self, X_val, y_val, hard_neg_mining=False):
        self.model.eval()
        mean_loss, step = 0, 0

        if not hard_neg_mining:
            inputs, targets = random_nonzero_crops(X_val, y_val,
                                                   self.num_of_patches // 4,
                                                   shape=self.crops_shape)
            batch_size_ = self.batch_size
        else:
            batch_size_ = 1
            train_losses = []
            inputs, targets = X_val, y_val
        #
        bar = tqdm(iterate_minibatches(inputs, targets, batch_size_, False),
                   total=inputs.shape[0] // self.batch_size)
        for x, y_true in bar:
            batch_loss_val = stochastic_step(x, y_true, self.model,
                                             self.optimizer, train=False)
            bar.set_description(str(batch_loss_val)[:6])
            mean_loss += batch_loss_val
            step += 1
            if hard_neg_mining:
                train_losses.append(batch_loss_val)
        if hard_neg_mining:
            hard_neg_idx = np.argsort(np.array(train_losses))[::-1][:30]
            self.neg_minig_storage = [inputs[hard_neg_idx], targets[hard_neg_idx]]
        return mean_loss / step


    def train(self, X_train, y_train, epoches=50,
              X_val=None, y_val=None, verbose=True):
        if not self.initialized:
            self.init_train_procedure()
        for epoch in range(epoches):
            tr_loss = self._train(X_train, y_train, hard_neg_mining=epoch>=1 and epoch%10!=0)
            if verbose:
                print(str(epoch)+' epoch: \ntrain loss', tr_loss)
            if not X_val is None:
                val_loss = self._validate(X_val, y_val)
                self._add_statistics(**{'val_loss': val_loss})
                if verbose:
                    print('val loss', val_loss)
            if epoch % 8 == 0:
                self.lr = self.lr_decayer(self.lr)
                self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.lr)
            self._add_statistics(**{'train_loss': tr_loss, 'lr': self.lr})
#endregion