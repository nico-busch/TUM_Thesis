import torch


def weighted_bce_loss(logits, targets, weights):
    loss = 0
    for f in range(len(logits)):
        loss += (logits[f].softmax(dim=1) * weights[f]).mean()
    return loss / len(logits)
