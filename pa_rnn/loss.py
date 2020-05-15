import torch


def cw_nll_loss(logits, weights):
    loss = 0
    for p in range(len(logits)):
        targets = torch.nn.functional.one_hot(weights[p].argmin(dim=1), weights[p].shape[1])
        loss += -(torch.log_softmax(logits[p], dim=1) * weights[p]).mean()
    return loss