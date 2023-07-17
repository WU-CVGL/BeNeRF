import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm, trange
import os
import imageio

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
img2se = lambda x, y: (x - y) ** 2
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))  # logab = logcb / logca
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
to8b_tensor = lambda x: (255 * torch.clip(x, 0, 1)).type(torch.int)


def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)


def load_imgs(path):
    imgfiles = [os.path.join(path, f) for f in sorted(os.listdir(path)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    imgs = imgs.astype(np.float32)
    imgs = torch.tensor(imgs).cuda()

    return imgs


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()  # i: [768, 480] value [[n*0],[n*1],...,[n*768]]
    j = j.t()  # j: [768, 480] 每列值相等
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


# Ray helpers only get specific rays
def get_specific_rays(i, j, K, c2w):
    # i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
    #                       torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    # i = i.t()
    # j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[..., :3, :3], -1)  # 每一个坐标对应一个 Rotation matrix
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

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def render_video_test(i_, graph, render_poses, H, W, K, args):
    rgbs = []
    disps = []
    # t = time.time()
    for i, pose in enumerate(tqdm(render_poses)):
        # print(i, time.time() - t)
        # t = time.time()
        pose = pose[None, :3, :4]
        ret = graph.render_video(i_, pose[:3, :4], H, W, K, args)  # 直接调用graph.render 可以在render加个设置， if ray_idx is None
        rgbs.append(ret['rgb_map'].cpu().numpy())
        disps.append(ret['disp_map'].cpu().numpy())
        if i == 0:
            print(ret['rgb_map'].shape, ret['disp_map'].shape)
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def render_image_test(i, graph, render_poses, H, W, K, args, dir=None, need_depth=True):
    img_dir = os.path.join(args.basedir, args.expname, dir, 'img_test_{:06d}'.format(i))
    os.makedirs(img_dir, exist_ok=True)
    imgs = []

    for j, pose in enumerate(tqdm(render_poses)):
        # print(i, time.time() - t)
        # t = time.time()
        pose = pose[None, :3, :4]
        ret = graph.render_video(i, pose[:3, :4], H, W, K, args)
        imgs.append(ret['rgb_map'])
        rgbs = ret['rgb_map'].cpu().numpy()
        rgb8 = to8b(rgbs)
        imageio.imwrite(os.path.join(img_dir, dir[11:] + '_{:03d}.png'.format(j)), rgb8)
        # imageio.imwrite(os.path.join(img_dir, 'rgb_{:03d}.png'.format(j)), rgb8)
        if need_depth:
            depths = ret['disp_map'].cpu().numpy()
            depths_ = depths / np.max(depths)
            depth8 = to8b(depths_)
            imageio.imwrite(os.path.join(img_dir, 'depth_{:03d}.png'.format(j)), depth8)
    imgs = torch.stack(imgs, 0)
    return imgs


def render_rolling_shutter_(barf_i, graph, render_poses, H, W, K, args, dir=None, need_depth=False):
    img_dir = os.path.join(args.basedir, args.expname, dir, 'img_test_{:06d}'.format(barf_i))
    os.makedirs(img_dir, exist_ok=True)
    imgs = []

    for i in range(render_poses.shape[0] // H):
        pose = render_poses[i * H: (i + 1) * H, :3, :4].unsqueeze(1).repeat(1, W, 1, 1).reshape(-1, 3, 4)
        ray_idx = torch.arange(H * W)
        img = []
        for j in range(H):
            ret = graph.render(barf_i, pose[j * W: (j + 1) * W, ...], ray_idx[j * W: (j + 1) * W, ...], H, W, K, args,
                               ray_idx_tv=None, training=True)
            img.append(ret['rgb_map'])
        imgs.append(torch.stack(img, 0))
        rgbs = torch.stack(img, 0).cpu().numpy()
        rgb8 = to8b(rgbs)
        imageio.imwrite(os.path.join(img_dir, dir[11:] + '_{:03d}.png'.format(i)), rgb8)
    imgs = torch.stack(imgs, 0)
    return imgs


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


if __name__ == '__main__':
    for i in range(10):
        ray_idx = compute_ray_idx(5, 6, 6)
        print(ray_idx.reshape(5, 5))
