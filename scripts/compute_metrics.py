import os.path

import numpy as np
import torch

from metrics import compute_img_metric
from imageio.v3 import imread, imwrite
from utils.imgutils import to8bit

channels = 1

if __name__ == '__main__':
    img = os.path.expanduser("C:\\Users\\User\\Desktop\\TestMetrics\\EDI\\img.png")
    gt = os.path.expanduser("C:\\Users\\User\\Desktop\\TestMetrics\\EDI\\gt.png")
    out = os.path.expanduser("C:\\Users\\User\\Desktop\\TestMetrics\\EDI\\out.png")
    txt = os.path.expanduser("C:\\Users\\User\\Desktop\\TestMetrics\\EDI\\out.txt")

    # img = os.path.expanduser("C:\\Users\\User\\Desktop\\TestMetrics\\SRN-Deblur\\img.png")
    # gt = os.path.expanduser("C:\\Users\\User\\Desktop\\TestMetrics\\SRN-Deblur\\gt.png")
    # out = os.path.expanduser("C:\\Users\\User\\Desktop\\TestMetrics\\SRN-Deblur\\out.png")
    # txt = os.path.expanduser("C:\\Users\\User\\Desktop\\TestMetrics\\SRN-Deblur\\out.txt")

    # img = os.path.expanduser("C:\\Users\\User\\Desktop\\TestMetrics\\img.png")
    # gt = os.path.expanduser("C:\\Users\\User\\Desktop\\TestMetrics\\gt.png")
    # out = os.path.expanduser("C:\\Users\\User\\Desktop\\TestMetrics\\out.png")
    # txt = os.path.expanduser("C:\\Users\\User\\Desktop\\TestMetrics\\out.txt")

    # read images
    if channels == 1:
        img = np.expand_dims(imread(img, mode="L") / 255., axis=(0, -1)).astype(np.float32)
        gt = np.expand_dims(imread(gt, mode="L") / 255., axis=(0, -1)).astype(np.float32)
    else:
        img = np.expand_dims(imread(img, mode="RGB")[...:3] / 255., axis=0).astype(np.float32)
        gt = np.expand_dims(imread(gt, mode="RGB")[...:3] / 255., axis=0).astype(np.float32)

    img_tensor = torch.tensor(img)
    gt_tensor = torch.tensor(gt)

    test_mid_mse = compute_img_metric(img_tensor, gt_tensor, metric="mse")
    test_mid_psnr = compute_img_metric(img_tensor, gt_tensor, metric="psnr")
    test_mid_ssim = compute_img_metric(img_tensor, gt_tensor, metric="ssim")
    test_mid_lpips = compute_img_metric(img_tensor, gt_tensor, metric="lpips")

    print(f"PSNR: {test_mid_psnr}")
    print(f"SSIM: {test_mid_ssim}")
    print(f"LPIPS: {test_mid_lpips}")

    diff = np.abs(img - gt).squeeze()

    imwrite(out, to8bit(diff), mode="L" if channels == 1 else "RGB")

    # write into txt
    file = open(txt, "w")
    file.write(f"PSNR: {test_mid_psnr}")
    file.write(f"SSIM: {test_mid_ssim}")
    file.write(f"LPIPS: {test_mid_lpips}")
    file.close()

    print("Finished")
