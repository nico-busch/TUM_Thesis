import torch


def pure_policy(mat, n_time):
    return [torch.full([n_time], mat).long()]
