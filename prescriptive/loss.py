import torch


def weighted_bce_loss(logits, targets, weights):
    loss = (-weights * (torch.nn.functional.logsigmoid(logits) * targets +
                        torch.nn.functional.logsigmoid(-logits) * (1 - targets)))
    return loss.mean()
