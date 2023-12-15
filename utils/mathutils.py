import torch

def safelog(x, eps=1e-12):
    x[..., 0:1]
    dim = x.size()
    eps = torch.Tensor([eps]).expand(dim)
    x = torch.max(x, torch.Tensor(eps))
    return torch.log(x)

