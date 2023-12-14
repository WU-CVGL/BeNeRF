import torch

def safelog(x, eps=1e-12):
    x = max(x, eps)
    return torch.log(x)

