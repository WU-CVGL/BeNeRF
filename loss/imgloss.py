import torch


class MSELoss:
    def __call__(self, img, gt_img):
        return torch.mean((img - gt_img) ** 2)
