import numpy as np
import torch
from imageio.v3 import imread
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


def to8bit(x) -> np.ndarray:
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def rgb2gray(x) -> np.ndarray:
    weight = np.array((0.299, 0.587, 0.114))
    x = np.sum(x * weight, axis=-1)
    x = x.astype(np.uint8)
    return x

# read image and normlization
def load_image(img, gray) -> np.ndarray:
    return (imread(img) / 255.).astype(np.float64) if gray else (imread(img)[..., :3] / 255.).astype(np.float64)
