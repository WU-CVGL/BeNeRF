import os

import numpy as np
import torch
from imageio.v3 import imwrite
from tqdm import tqdm

from utils import imgutils

tonemap = lambda x : (np.log(np.clip(x,0,1) * 5000 + 1 ) / np.log(5000 + 1) * 255).astype(np.uint8)

# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


# Ray helpers only get specific rays
def get_specific_rays(i, j, K, c2w):
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[..., :3, :3], -1)
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[..., :3, -1]
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def render_video_test(graph, render_poses, H, W, K, args):
    rgbs = []
    disps = []
    radiences = []
    for i, pose in enumerate(tqdm(render_poses)):
        pose = pose[None, :3, :4]
        ret = graph.render_video(pose[:3, :4], H, W, K, args, type = "rgb")
        ret_radience = graph.render_video(pose[:3, :4], H, W, K, args, type = "radience")
        rgbs.append(ret['rgb_map'].cpu().numpy())
        disps.append(ret['disp_map'].cpu().numpy())

        radience = ret_radience['rgb_map'].cpu().numpy()
        radience = tonemap(radience / np.max(radience))
        radiences.append(radience)

        if i == 0:
            print(ret['rgb_map'].shape, ret['disp_map'].shape)
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    radiences = np.stack(radiences, 0)
    return rgbs, radiences, disps


def render_image_test(i, graph, render_poses, H, W, K, args, logdir, dir=None, need_depth=True):
    img_dir = os.path.join(logdir, dir, 'img_test_{:06d}'.format(i))
    os.makedirs(img_dir, exist_ok=True)
    imgs = []
    radiences = []
    depth = []

    for j, pose in enumerate(tqdm(render_poses)):
        pose = pose[None, :3, :4]
        ret = graph.render_video(pose[:3, :4], H, W, K, args, type = "rgb")
        ret_radience = graph.render_video(pose[:3, :4], H, W, K, args, type = "radience")
        rgbs = ret['rgb_map'].cpu().numpy()
        radience = ret_radience['rgb_map'].cpu().numpy()
        rgb8 = imgutils.to8bit(rgbs)
        radience = tonemap(radience / np.max(radience))
        imwrite(os.path.join(img_dir, dir[11:] + 'img_{:03d}.png'.format(j)), rgb8.squeeze(),
                mode="L" if args.channels == 1 else "RGB")
        imwrite(os.path.join(img_dir, dir[11:] + 'radience_{:03d}.png'.format(j)), rgb8.squeeze(),
                mode="L" if args.channels == 1 else "RGB")
        imgs.append(rgb8)
        radiences.append(radience)
        if need_depth:
            depths = ret['disp_map'].cpu().numpy()
            depths_ = depths / np.max(depths)
            depth8 = imgutils.to8bit(depths_)
            imwrite(os.path.join(img_dir, 'depth_{:03d}.png'.format(j)), depth8)
            depth.append(depth8)
    return imgs, radiences, depth


def compute_poses_idx(img_idx, args):
    poses_idx = torch.arange(img_idx.shape[0] * args.deblur_images)
    for i in range(img_idx.shape[0]):
        for j in range(args.deblur_images):
            poses_idx[i * args.deblur_images + j] = img_idx[i] * args.deblur_images + j
    return poses_idx


def compute_ray_idx(width, H, W):
    ray_idx_start = torch.randint(H * W, (1,))
    while (ray_idx_start[0] % W > (W - width)) or (ray_idx_start[0] // H > (H - width)):
        ray_idx_start = torch.randint(H * W, (1,))

    ray_idx_list = []
    for h in range(width):  # height 480
        for w in range(width):  # width 768
            ray_idx_ = ray_idx_start + h * H + w
            ray_idx_list.append(ray_idx_)
    ray_idx = torch.stack(ray_idx_list)
    ray_idx = ray_idx.squeeze()
    return ray_idx

def init_weights(linear):
    # use Xavier init instead of Kaiming init
    torch.nn.init.xavier_uniform_(linear.weight)
    torch.nn.init.zeros_(linear.bias)

def init_nerf(nerf):
    for linear_pt in nerf.pts_linears:
        init_weights(linear_pt)

    for linear_view in nerf.views_linears:
        init_weights(linear_view)

    init_weights(nerf.feature_linear)
    init_weights(nerf.alpha_linear)
    init_weights(nerf.rgb_linear)
