import torch
from torch import Tensor


class RGB2Gray:
    def __init__(self) -> None:
        r = 0.299
        g = 0.587
        b = 0.114
        self.rgb_weight = torch.Tensor([r, g, b])

    def __call__(self, rgb) -> Tensor:
        x = torch.sum(rgb * self.rgb_weight[None, :], axis=-1)
        return x.reshape(x.shape[0], 1)
