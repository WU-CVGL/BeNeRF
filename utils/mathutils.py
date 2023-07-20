import torch


def safelog(x, eps=1e-4):
    return torch.log(x + eps)

