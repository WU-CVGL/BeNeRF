from metrics import compute_img_metric
import run_nerf_helpers
from diff_img import *
import os

imgs_render_dir = './logs/Synthetic-Datasets/Living-Room-1000Hz/test_poses_mid/img_test_010000'
imgs_sharp_dir = './Data/Synthetic-Datasets/Living-Room-1000Hz/images'

save_path = os.path.join(imgs_render_dir, 'diff_res')

imgs_sharp = run_nerf_helpers.load_imgs(imgs_sharp_dir)
imgs_render = run_nerf_helpers.load_imgs(imgs_render_dir)

imgs_gray_render = (imgs_render[..., 0].cpu().numpy() * 255).astype(np.uint8)

for i in range(imgs_gray_render.shape[0]):
    dir = os.path.join(imgs_render_dir, '{:03d}_gray.png'.format(i))
    imageio.imwrite(dir, imgs_gray_render[i])

exit()

test_metric_file = os.path.join(imgs_render_dir, 'test_metrics.txt')
open(test_metric_file, "w")

for i in range(imgs_sharp.shape[0]):
    mse_render = compute_img_metric(imgs_sharp[[i]], imgs_render[[i]], 'mse')
    psnr_render = compute_img_metric(imgs_sharp[[i]], imgs_render[[i]], 'psnr')
    ssim_render = compute_img_metric(imgs_sharp[[i]], imgs_render[[i]], 'ssim')
    lpips_render = compute_img_metric(imgs_sharp[[i]], imgs_render[[i]], 'lpips')

    with open(test_metric_file, 'a') as outfile:
        outfile.write(f"Image{i}: MSE:{mse_render.item():.8f} PSNR:{psnr_render.item():.8f}"
                      f" SSIM:{ssim_render.item():.8f} LPIPS:{lpips_render.item():.8f}\n")
        mse_render = compute_img_metric(imgs_sharp, imgs_render, 'mse')
        psnr_render = compute_img_metric(imgs_sharp, imgs_render, 'psnr')
        ssim_render = compute_img_metric(imgs_sharp, imgs_render, 'ssim')
        lpips_render = compute_img_metric(imgs_sharp, imgs_render, 'lpips')

with open(test_metric_file, 'a') as outfile:
    outfile.write(
        f"Average: MSE:{mse_render.item():.8f} PSNR:{psnr_render.item():.8f}"f" SSIM:{ssim_render.item():.8f} LPIPS:{lpips_render.item():.8f}\n")
img_diff(imgs_sharp.cpu().numpy(), imgs_render.cpu().numpy(), save_path)
