import os.path
import random

import configargparse
import numpy as np
import torch
from tqdm.contrib import tzip

from metrics import compute_img_metric
from imageio.v3 import imread, imwrite
from utils.imgutils import to8bit


def read_img(img, channels):
    if channels == 1:
        return np.expand_dims(imread(img, mode="L") / 255., axis=(0, -1)).astype(np.float32)
    else:
        return np.expand_dims(imread(img, mode="RGB")[...:3] / 255., axis=0).astype(np.float32)


if __name__ == '__main__':
    print("Starting metrics...")
    parser = configargparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default='../result')
    parser.add_argument("--gt_dir", type=str, default='../result')
    parser.add_argument("--out_dir", type=str, default='../result/output')
    parser.add_argument("--channels", type=int, default=1)
    args = parser.parse_args()

    print("Loading data...")
    img = os.path.expanduser(args.in_dir)
    gt = os.path.expanduser(args.gt_dir)
    out = os.path.expanduser(args.out_dir)
    metrics = os.path.join(out, "metrics.csv")
    result = os.path.join(out, "result.txt")
    channels = args.channels
    assert channels == 1 or channels == 3

    os.makedirs(img, exist_ok=True)
    os.makedirs(gt, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    img = [os.path.join(img, f) for f in sorted(os.listdir(img)) if f.lower().endswith(("jpg", "png"))]
    gt = [os.path.join(gt, f) for f in sorted(os.listdir(gt)) if f.lower().endswith(("jpg", "png"))]
    img = [read_img(f, channels) for f in img]
    gt = [read_img(f, channels) for f in gt]
    assert len(img) == len(gt)

    print("Computing metrics...")
    # setup seeds
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.random.manual_seed(0)
    random.seed(0)
    # compute
    avg_psnr, avg_ssim, avg_mse, avg_lpips = 0, 0, 0, 0
    metrics = open(metrics, "w")
    metrics.write("MSE, PSNR, SSIM, LPIPS\n")
    for i, g, idx in tzip(img, gt, range(len(img))):
        img_tensor = torch.tensor(i)
        gt_tensor = torch.tensor(g)

        test_mid_mse = compute_img_metric(img_tensor, gt_tensor, metric="mse")
        test_mid_psnr = compute_img_metric(img_tensor, gt_tensor, metric="psnr")
        test_mid_ssim = compute_img_metric(img_tensor, gt_tensor, metric="ssim")
        test_mid_lpips = compute_img_metric(img_tensor, gt_tensor, metric="lpips")

        # add avg
        avg_mse += test_mid_mse / len(gt)
        avg_psnr += test_mid_psnr / len(gt)
        avg_ssim += test_mid_ssim / len(gt)
        avg_lpips += test_mid_lpips / len(gt)

        print(f"{idx} -> MSE: {test_mid_mse:.4f} PSNR: {test_mid_psnr:.2f} SSIM: {test_mid_ssim:.2f} LPIPS: {test_mid_lpips:.3f}")

        # write into metrics
        metrics.write(f"{test_mid_mse:.4f}, {test_mid_psnr:.2f}, {test_mid_ssim:.2f}, {test_mid_lpips:.2f}\n")

        # compute diff
        diff = np.abs(i - g).squeeze()
        imwrite(os.path.join(out, f"{idx}.png"), to8bit(diff), mode="L" if channels == 1 else "RGB")

    metrics.close()

    print(f"AVG MSE: {avg_mse:.4f} AVG PSNR: {avg_psnr:.2f} AVG SSIM: {avg_ssim:.2f} AVG LPIPS: {avg_lpips:.3f}")
    with open(result, "w") as f:
        f.write(
            f"AVG MSE: {avg_mse:.4f}\nAVG PSNR: {avg_psnr:.2f}\nAVG SSIM: {avg_ssim:.2f}\nAVG LPIPS: {avg_lpips:.3f}")
    print("Finished")
