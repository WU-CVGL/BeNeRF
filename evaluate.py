import os
import torch
import metrics
import imageio
import argparse
import torch.hub
import numpy as np

def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, format="PNG-PIL", ignoregamma=True)
    else:
        return imageio.imread(f)

def load_imgs(path):
    path = os.path.expanduser(path)
    imgfiles = [os.path.join(path, f) for f in sorted(os.listdir(path)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    imgs = imgs.astype(np.float32)
    imgs = torch.tensor(imgs).cuda()
    return imgs

def evaluate(sharp_path, deblur_path):
    sharp_img = load_imgs(sharp_path)
    deblur_img = load_imgs(deblur_path)

    deblur_psnr = metrics.compute_img_metric(deblur_img, sharp_img, 'psnr')
    deblur_ssim = metrics.compute_img_metric(deblur_img, sharp_img, 'ssim')
    deblur_lpips = metrics.compute_img_metric(deblur_img, sharp_img, 'lpips')
    if isinstance(deblur_lpips, torch.Tensor):
        deblur_lpips = deblur_lpips.item()
    return deblur_psnr, deblur_ssim, deblur_lpips

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Name of dataset')
    parser.add_argument('--scene', type=str, help='Name of scene')
    parser.add_argument('--result', type=str, help='Path to the directory of results')
    parser.add_argument('--groundtruth', type=str, help='Path to the directory of groundtruth')
    args = parser.parse_args()

    psnr_list = []
    ssim_list = []
    lpips_list = []

    deblur_psnr, deblur_ssim, deblur_lpips = evaluate(args.result, args.groundtruth)
    psnr_list.append(deblur_psnr)
    ssim_list.append(deblur_ssim)
    lpips_list.append(deblur_lpips)
    print("******************{}******************".format(args.dataset+"_"+args.scene))
    print('psnr: high||', deblur_psnr)
    print('ssim: high||', deblur_ssim)
    print('lpips:low ||', deblur_lpips)
