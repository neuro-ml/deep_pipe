from torch.nn import functional


def focal_loss_with_logits(logits, target, gamma=2, weight=None, size_average=True, reduce=True):
    """
    Function that measures Focal Loss between target and output logits.

    Parameters
    ----------
    logits: Variable of arbitrary shape
    target: Variable of the same shape as input
    gamma: float
        The power of focal loss factor
    weight: Variable, optional
        a manual rescaling weight. If provided it's repeated to match input tensor shape
    size_average: bool, optional
        By default, the losses are averaged over observations for each minibatch. However, if the field
        :attr:`size_average` is set to ``False``, the losses are instead summed
        for each minibatch. Default: ``True``
    reduce: bool, optional
        By default, the losses are averaged or summed over
        observations for each minibatch depending on :attr:`size_average`. When :attr:`reduce`
        is ``False``, returns a loss per logits/target element instead and ignores
        :attr:`size_average`. Default: ``True``
    """
    if not (target.size() == logits.size()):
        raise ValueError("Target size ({}) must be the same as logits size ({})".format(target.size(), logits.size()))

    min_val = - logits.clamp(min=0)
    max_val = (-logits).clamp(min=0)

    prob = (min_val + logits).exp() / (min_val.exp() + (min_val + logits).exp())

    loss = ((1 - 2 * prob) * target + prob) ** gamma * (
            logits - logits * target + max_val + ((-max_val).exp() + (-logits - max_val).exp()).log())

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


def linear_focal_loss_with_logits(logits, target, gamma, beta, weight=None, size_average=True, reduce=True):
    return functional.binary_cross_entropy_with_logits(
        gamma * logits + beta, target, weight, size_average, reduce
    ) / gamma
