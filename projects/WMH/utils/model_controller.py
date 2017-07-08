import torch
import numpy as np
from tqdm import tqdm

from data_utils import random_nonzero_crops, random_slicer, divide, combine, \
    augment
from data_utils import _reshape_to, divider, combiner
from nn_utils import iterate_minibatches
from pytorch_utils import stochastic_step, to_numpy, to_var
from mulptiprocessing_utils import par_iterate_minibatches


class Model_controller():
    """
    Model controller - incapsulate train/val methods for NN model.

    --------
    Example how to use:
    model = UNet().cuda() # PyTorch
    controller = Model_controller(model)
    controller.init_train_procedure() #default params
    controller.train(X_train, Y_train, epoches=100,
                     hard_negative = lambda epoch: epoch>=1 and epoch%7!=0)
    y = controller.predict(X_test)
    np.savez('y_dump.npy', y)
    """

    def __init__(self, model):
        self.model = model

    def init_train_procedure(self, batch_size=20, lr=0.01, net='3D',
                             num_of_patches=500, crops_shape=(52, 52, 44),
                             context_shape=(104, 104, 88),
                             lr_decayer=lambda x: x * 0.5,
                             decay_every_epoch=20):
        """
        Init params.
        """
        assert net == '2D' or net == '3D'
        self.initialized = True
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.num_of_patches = num_of_patches
        self.crops_shape = crops_shape
        self.context_shape = context_shape
        self.lr_decayer = lr_decayer
        self.decay_every_epoch = decay_every_epoch
        self.res = dict()
        self.net_dim = net
        self.neg_minig_storage = []

    def _add_statistics(self, **kwargs):
        for name in kwargs:
            stat_vars = self.res.keys()
            if name in stat_vars:
                self.res[name].append(kwargs[name])
            else:
                self.res[name] = []

    def _train(self, X_train, y_train, hard_neg_mining=False):
        self.model.train()
        mean_loss, step = 0, 0
        if self.net_dim == '3D':
            inputs, targets = random_nonzero_crops(X_train, y_train,
                                                   self.num_of_patches,
                                                   targ_shape=self.context_shape,
                                                   context_shape=self.crops_shape)
        else:
            inputs, targets = random_slicer(X_train, y_train,
                                            num_of_slices=self.num_of_patches)
        if hard_neg_mining and len(self.neg_minig_storage) > 0:
            prev_inp = self.neg_minig_storage[0]
            prev_targ = self.neg_minig_storage[1]
            print('added', len(prev_inp), 'examples')
            inputs = np.concatenate([inputs, prev_inp])
            targets = np.concatenate([targets, prev_targ])

        bar = tqdm(
            par_iterate_minibatches(inputs, targets, self.batch_size, augment),
            total=inputs.shape[0] // self.batch_size)
        for x, y_true in bar:
            batch_loss_val = stochastic_step(x, y_true, self.model,
                                             self.optimizer)
            bar.set_description(str(batch_loss_val)[:6])
            mean_loss += batch_loss_val
            step += 1
        if hard_neg_mining:
            _ = self._validate(inputs, targets, hard_neg_mining=True)
        return mean_loss / step

    def _validate(self, X_val, y_val, hard_neg_mining=False):
        self.model.eval()
        mean_loss, step = 0, 0
        train_losses = []  # for hard negative mining scores.
        if hard_neg_mining:
            batch_size_ = 1
            inputs, targets = X_val, y_val
        else:
            if self.net_dim == '3D':
                inputs, targets = random_nonzero_crops(X_val, y_val,
                                                       self.num_of_patches // 4,
                                                       targ_shape=self.crops_shape,
                                                       context_shape=self.context_shape)
            else:
                inputs, targets = random_slicer(X_val, y_val,
                                                num_of_slices=self.num_of_patches // 4)
            batch_size_ = self.batch_size

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
            self.neg_minig_storage = [inputs[hard_neg_idx],
                                      targets[hard_neg_idx]]
        return mean_loss / step

    def train(self, X_train, y_train, epoches=50, X_val=None, y_val=None,
              verbose=True, hard_negative=lambda epoch: epoch >= 1):
        if not self.initialized:
            self.init_train_procedure()
        for epoch in range(epoches):
            tr_loss = self._train(X_train, y_train,
                                  hard_neg_mining=hard_negative(epoch))
            self._add_statistics(**{'train_loss': tr_loss})
            if verbose:
                print(str(epoch) + ' epoch: \ntrain loss', tr_loss)
            if not X_val is None:
                val_loss = self._validate(X_val, y_val)
                self._add_statistics(**{'val_loss': val_loss})
                if verbose:
                    print('val loss', val_loss)
            if epoch % self.decay_every_epoch == 0:
                self.lr = self.lr_decayer(self.lr)
                # TODO not only pytorch
                self.optimizer = torch.optim.Adamax(self.model.parameters(),
                                                    lr=self.lr)
            for i in self.optimizer.param_groups:
                self._add_statistics(**{'lr': i['lr']})
                break

    def predict(self, X, zero_padding=[0] * 5,
                n_parts_per_axis=(1, 1, 4, 4, 2)):
        """
        Divide X into n_parts_per_axis with padding.
        """
        shape = X.shape
        if self.net_dim == '3D':
            a_parts = divide(X, zero_padding, n_parts_per_axis)
        else:
            a_parts = X.reshape(shape[0] * shape[-1], shape[1], shape[2],
                                shape[3])
        pred = []
        # TODO implementation not only for PyTorch
        self.model.eval()
        for i in a_parts:
            if self.net_dim == '2D':
                i = i[np.newaxis]
            i = to_var(i).cuda()
            pred.append(to_numpy(self.model(i)))

        if self.net_dim == '3D':
            predictions = combine(pred, n_parts_per_axis)
        else:
            shape_ = list(shape)
            shape_[1:4] = list(np.array(pred).shape)[2:5]  #

            predictions = np.array(pred).reshape(*shape_)
        return predictions

    def common_predict(self, X, targ_shape, context_shape):
        """
        For data_utils.divider / combiner approach.
        """
        X_ = _reshape_to(X, np.array(X.shape) + np.array(
            [0, 0, *np.array(context_shape[2:])]))
        context, target = divider(X_.shape, targ_shape, context_shape)
        pred = []
        self.model.eval()
        for i, j in zip(context, target):
            i = to_var(X_[i]).cuda()
            pred.append(to_numpy(self.model(i)))
        pred = np.array(pred)
        return combiner(pred, target, X.shape)