import torch
from torch.autograd import Variable

from dpipe.config import register
from .model import Model, get_model_path


@register('torch', 'model')
class TorchModel(Model):
    def __init__(self, model_core: torch.nn.Module, logits2pred: callable, logits2loss: callable,
                 optimize: torch.optim.Optimizer, cuda=True):
        if cuda:
            model_core.cuda()
        self.cuda = cuda
        self.model_core = model_core
        self.logits2pred = logits2pred
        self.logits2loss = logits2loss
        self.optimize = optimize

    def do_train_step(self, *inputs, target, lr):
        self.model_core.train()
        inputs = [to_var(x, self.cuda) for x in inputs]
        target = to_var(target, self.cuda)

        output = self.model(*inputs)
        loss = self.logits2loss(self.logits2pred(output), target)

        set_lr(self.optimizer, lr)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        return to_np(loss)

    def do_val_step(self, *inputs, target):
        self.model_core.eval()
        inputs = [to_var(x, self.cuda) for x in inputs]
        target = to_var(target, self.cuda)

        output = self.model(*inputs)
        loss = self.logits2loss(self.logits2pred(output), target)

        return to_np(output), to_np(loss)

    def do_inf_step(self, *inputs):
        self.model_core.eval()
        inputs = [to_var(x, self.cuda) for x in inputs]
        output = self.model(*inputs)

        return to_np(output)

    def save(self, path):
        self.model_core.save_state_dict(get_model_path(path))

    def load(self, path):
        path = get_model_path(path)
        self.model_core.load_state_dict(torch.load(path))


@register('torch', 'frozen_model')
class FrozenModel:
    def __init__(self, model_core: torch.nn.Module, logits2pred: callable, restore_model_path, cuda=True):
        if cuda:
            model_core.cuda()
        self.cuda = cuda
        self.model_core = model_core
        self.logits2pred = logits2pred

        path = get_model_path(restore_model_path)
        self.model_core.load_state_dict(torch.load(path))

    def do_inf_step(self, *inputs):
        self.model_core.eval()
        inputs = [to_var(x, self.cuda) for x in inputs]
        output = self.model(*inputs)

        return to_np(output)


def to_np(x: Variable):
    return x.cpu().data.numpy()


def to_var(x, cuda=None):
    x = Variable(torch.from_numpy(x))
    if (torch.cuda.is_available() and cuda is None) or cuda:
        x = x.cuda()
    return x


def set_lr(optimizer: torch.optim.Optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
