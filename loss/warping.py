import numpy as np
import torch
from einops import rearrange

s = 1.


def to_homogeneous(X):
    if X.shape[-1] == 1:
        return torch.cat([X, torch.ones_like(X)[..., :1, :]], dim=-2)
    else:
        return torch.cat([X, torch.ones_like(X)[..., :1]], dim=-1)


def pix_loc_src_to_tgt(uv, intrin_src, c2w_src, intrin_tgt, c2w_tgt, depth_src):
    """
    Mapping the location of pixels from source view to target view
    params:
        uv: pixel coordinates, (B, H*W, 2)
        intrin: list[fx, fy, cx, cy]
        c2w_src, c2w_tgt: (B, 4, 4)
        depth_src: the depth values of all pixels under source view, (B, H*W)
    """
    fx, fy, cx, cy = intrin_src
    fx_t, fy_t, cx_t, cy_t = intrin_tgt
    device = c2w_src.device

    K = torch.tensor([
        [fx_t, 0, cx_t],
        [0, fy_t, cy_t],
        [0, 0, 1]
    ]).float().to(device=device)

    K_inv = torch.tensor([
        [1 / fx, 0, -cx / fx],
        [0, 1 / fy, -cy / fy],
        [0, 0, 1]
    ]).float().to(device=device)

    uv = to_homogeneous(uv).unsqueeze(-1)  # (B, N, 3, 1)
    X_c = torch.einsum('ij, bnjk -> bnik', K_inv, uv)
    X_c *= depth_src.reshape(-1, 1, 1)
    X_c = to_homogeneous(X_c)  # (B, N, 4, 1)
    X_w = torch.einsum('bij, bnjk -> bnik', c2w_src, X_c)
    X_c_ = torch.einsum('bij, bnjk -> bnik', torch.linalg.inv(c2w_tgt), X_w)
    X_c_ = X_c_[..., :3, :] / X_c_[..., [2], :]
    uv_ = torch.einsum('ij, bnjk -> bnik', K, X_c_)
    uv_ = uv_.squeeze(-1)[..., :2]  # (B, N, 2)

    return uv_  # notice that those pixel locations are not ceiled


if __name__ == '__main__':
    ray_idx_event = torch.randperm(680 * 480)[:1500]
    uv = torch.vstack((torch.tensor((1.)), torch.tensor(2.,))).t().unsqueeze(0).float()
    depth_src = torch.rand(1, 1)
    intrin = [500, 500, 200, 200]
    intrin = [1000, 1000, 300, 300]
    c2w_src = torch.rand((1, 4, 4))
    c2w_tgt = torch.rand((1, 4, 4))
    result = pix_loc_src_to_tgt(uv, intrin, c2w_src, intrin, c2w_tgt, depth_src)
    result = torch.round(result).to(torch.int)
    mask = (0 <= result[..., 0]) & (result[..., 0] < 680) & (0 <= result[..., 1]) & (result[..., 1] < 480)
    result = result[mask]
    torch.prod(result, dim=1)
    print(result)


def image_forward_warping(image, c2w_src, c2w_tgt, intrin, depth_values):
    MAX_DEPTH = depth_values.max()

    device = c2w_src.device
    # prepare image
    if image.dtype == np.uint8:
        image = torch.from_numpy(image).float()
        image = image / 127.5 - 1.0
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.shape[1] == 3:
        image = rearrange(image, 'b c h w -> b h w c')

    # apply image warping
    fx, fy, cx, cy = intrin
    B, H, W, C = image.shape

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device),
                          indexing='ij')
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    uv = torch.cat([i[..., None], j[..., None]], dim=-1)

    depth = depth_values.reshape(B, H * W)

    image_warpped = -torch.ones_like(image)
    image = rearrange(image, 'b h w c -> b (h w) c')

    for b in range(B):
        depth_b = depth[b]
        mask_b = depth_b != depth_b.max()
        depth_b = depth_b[mask_b]
        uv_b = uv[b][mask_b].unsqueeze(0)

        uv_tgt = pix_loc_src_to_tgt(uv_b, intrin, c2w_src[[b], ...], c2w_tgt[[b], ...], depth_b)
        uv_tgt = (uv_tgt[0] - 0.5).ceil().long()
        xs_pix_tgt, ys_pix_tgt = uv_tgt[..., 0], uv_tgt[..., 1]

        mask_tgt = (xs_pix_tgt < W) & \
                   (xs_pix_tgt >= 0) & \
                   (ys_pix_tgt < H) & \
                   (ys_pix_tgt >= 0)
        xs, ys = xs_pix_tgt[mask_tgt], ys_pix_tgt[mask_tgt]
        pixel_values = image[b][mask_b][mask_tgt]
        image_warpped[b, ys, xs, :] = pixel_values

    return image_warpped


def image_backward_warping(image_src, c2w_src, image_tgt, c2w_tgt, intrin, depth_src):
    """
    Perform inverse warping
    params:
        image_src, image_tgt: torch.Tensor, (B, C, H, W), values in (-1, 1)
        c2w_src, c2w_tgt: (B, 4, 4)
    
    """
    MAX_DEPTH = depth_src.max()

    device = c2w_src.device

    # apply image warping
    fx, fy, cx, cy = intrin
    B, C, H, W = image_tgt.shape

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device),
                          indexing='ij')
    i = i.t().reshape([1, H * W]).expand([B, H * W])
    j = j.t().reshape([1, H * W]).expand([B, H * W])
    uv = torch.cat([i[..., None], j[..., None]], dim=-1)
    depth = depth_src.reshape(B, H * W)

    uv_tgt = pix_loc_src_to_tgt(uv + 0.5, intrin, c2w_src, c2w_tgt, depth)  # (B, H*W, 2)
    # uv_tgt = rearrange(uv_tgt, "b (h w) k -> b h w k", h=H)
    uv_tgt = uv_tgt / torch.tensor([W / 2, H / 2], device=device) - 1
    # xs_pix_tgt, ys_pix_tgt = uv_tgt[..., 0], uv_tgt[..., 1]
    mask = depth != MAX_DEPTH
    image_src_warpped = -torch.ones_like(image_src)

    for i in range(B):
        mask_ = mask[i]
        uv_src = uv[i][mask_].long()
        uv_ = uv_tgt[i][mask_]
        uv_ = uv_.reshape(1, 1, *uv_.shape).to(dtype=image_tgt.dtype)
        pixel_values = torch.nn.functional.grid_sample(
            image_tgt[[i]], uv_, mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze().to(dtype=image_src.dtype)  # (3, N_mask)
        image_src_warpped[i, :, uv_src[:, 1], uv_src[:, 0]] = pixel_values

    # image_src_warpped = torch.nn.functional.grid_sample(
    #     image_tgt, uv_tgt, mode='bilinear', padding_mode='border', align_corners=False
    # )

    return image_src_warpped


def proj_pix_to_world(image, c2w, intrin, depth_values):
    MAX_DEPTH = depth_values.max()
    device = c2w.device
    # prepare image
    if image.dtype == np.uint8:
        image = torch.from_numpy(image).float()
        image = image / 127.5 - 1.0
    assert image.ndim == 3

    # apply image warping
    fx, fy, cx, cy = intrin
    H, W, C = image.shape

    xx, yy = torch.meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device),
                            indexing='ij')
    xx = xx.t().flatten() + 0.5
    yy = yy.t().flatten() + 0.5

    fx, fy, cx, cy = intrin
    depth = depth_values.flatten()
    xs = (xx - cx) / fx * s * depth
    ys = (yy - cy) / fy * s * depth
    zs = torch.ones_like(xs) * s * depth
    X_c = torch.cat(
        [xs[..., None], ys[..., None], zs[..., None],
         torch.ones((H * W, 1), device=device)], dim=-1
    ).unsqueeze(-1)  # (H*W, 4, 1)
    X_w = torch.einsum('ij, njk -> nik', c2w, X_c)
    X_w = X_w[:, :3, 0]  # (N, 3)

    mask = depth != MAX_DEPTH

    points = X_w[mask]

    return points
