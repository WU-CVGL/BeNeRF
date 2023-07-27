import torch


def safelog(x, eps=1e-3):
    return torch.log(x + eps)

