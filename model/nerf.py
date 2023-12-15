import abc
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from model import embedder
from run_nerf_helpers import get_specific_rays, get_rays, ndc_rays, sample_pdf
from utils.eventutils import accumulate_events

class Model:
    @abc.abstractmethod
    def build_network(self, args, poses=None, event_poses=None):
        pass

    @abc.abstractmethod
    def setup_optimizer(self, args):
        pass

    def after_train(self):
        print(f"Successfully finished model on {datetime.now()}")

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=False,
                 channels=3):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.channels = channels

        # network
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, channels)
        else:
            self.output_linear = nn.Linear(W, channels + 1)

    # positional encoding和nerf的mlp
    def forward(self, pts, viewdirs, args):
        # create positional encoding
        embed_fn, input_ch = embedder.get_embedder(args.multires, args.i_embed)
        input_ch_views = 0
        embeddirs_fn = None
        if args.use_viewdirs:
            embeddirs_fn, input_ch_views = embedder.get_embedder(args.multires_views, args.i_embed)
        # forward positional encoding
        pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])
        embedded = embed_fn(pts_flat)

        if viewdirs is not None:
            # embedded_dirs:[1024x64, 27]
            input_dirs = viewdirs[:, None].expand(pts.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        input_pts, input_views = torch.split(embedded, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)  # [N, 4(RGBA A is sigma)]
        else:
            outputs = self.output_linear(h)

        outputs = torch.reshape(outputs, list(pts.shape[:-1]) + [outputs.shape[-1]])

        return outputs

    def raw2output(self, crf, sensor_type, raw, z_vals, rays_d, raw_noise_std=1.0):
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :self.channels])

        rgb = crf(rgb, sensor_type)

        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., self.channels].shape) * raw_noise_std

        alpha = raw2alpha(raw[..., self.channels] + noise, dists)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,
                          :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        sigma = F.relu(raw[..., self.channels] + noise)

        return rgb_map, disp_map, acc_map, weights, depth_map, sigma

class Graph(nn.Module):

    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=False):
        super().__init__()
        self.nerf = NeRF(D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs, args.channels)
        self.channels = args.channels
        if args.N_importance > 0:
            self.nerf_fine = NeRF(D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs, args.channels)
        self.pose_eye = torch.eye(3, 4)

    def forward(self, i, events, H, W, K, K_event, args):
        # select events in time windows(length: 0.1s)
        if args.time_window:
            window_t = args.window_percent
            if args.random_window:
                low_t = np.random.rand(1) * (1 - window_t)
                upper_t = low_t + window_t
            else:
                low_t = np.random.randint((1 - window_t) // window_t) * window_t
                upper_t = np.min((low_t + window_t, 1.0))
            idx_a = low_t <= events["ts"]
            idx_b = events["ts"] <= upper_t
            idx = idx_a * idx_b
            indices = np.where(idx)
            # get element in time window
            pol_window = events['pol'][indices]
            x_window = events['x'][indices]
            y_window = events['y'][indices]
            ts_window = events['ts'][indices]
        else:
            num = len(events["pol"])
            N_window = round(num * args.window_percent)
            if args.random_window:
                window_low_bound = np.random.randint(num - N_window)
                window_up_bound = int(window_low_bound + N_window)
            else:
                window_low_bound = np.random.randint((num - N_window) // N_window) * N_window
                window_up_bound = int(window_low_bound + N_window)
            pol_window = events['pol'][window_low_bound:window_up_bound]
            x_window = events['x'][window_low_bound:window_up_bound]
            y_window = events['y'][window_low_bound:window_up_bound]
            ts_window = events['ts'][window_low_bound:window_up_bound]

        # event temporal aggregate
        out = np.zeros((args.h_event, args.w_event))
        accumulate_events(out, x_window, y_window, pol_window)
        events_accu = torch.tensor(out)

        # timestamps of event windows begin and end
        if args.time_window:
            events_ts = np.stack((low_t, upper_t)).reshape(2)
        else:
            events_ts = ts_window[np.array([0, int(N_window) - 1])]
        
        # index of event rays
        ray_idx_event = torch.randperm(args.h_event * args.w_event)[:args.pix_event]
        # interpolated event pose 
        spline_poses = self.get_pose(args, torch.tensor(events_ts, dtype=torch.float32))
        # interpolated rgb pose 
        spline_rgb_poses = self.get_pose_rgb(args)

        # render event
        ret_event = self.render(spline_poses, 
                                ray_idx_event.reshape(-1, 1).squeeze(), 
                                args.h_event, 
                                args.w_event,
                                K_event,
                                args,
                                sensor_type = "event",
                                training=True)
        # warping loss
        if False:
            # get the pixel of event camera image
            uv = torch.vstack((ray_idx_event // args.w_event, ray_idx_event % args.w_event)).t().unsqueeze(0).float()
            # get the depth of event camera image
            depth_src = ret_event['disp_map'][:args.pix_event]
            # event camera intrin
            intrin = [args.focal_event_x, args.focal_event_y, args.event_cx, args.event_cy]
            # rgb camera intrin
            intrin_rgb = [args.focal_x, args.focal_y, args.cx, args.cy]
            from loss.warping import pix_loc_src_to_tgt
            # cal new pixel in rgb camera
            result = pix_loc_src_to_tgt(uv, intrin, spline_poses[1], intrin_rgb,
                                        spline_rgb_poses[spline_rgb_poses.shape[0] // 2], depth_src)
            result = torch.round(result).to(torch.int)
            # choose only the possible pixels
            mask = (0 <= result[..., 0]) & (result[..., 0] < W) & (0 <= result[..., 1]) & (result[..., 1] < H)
            result = result[mask]
            print(result.shape)
            additional_idx = torch.prod(result, dim=1)
            ret_warp = self.render(spline_rgb_poses[spline_rgb_poses.shape[0] // 2],
                                   additional_idx.reshape(-1, 1).squeeze(), H, W, K, args,
                                   training=True)

        # render rgb
        ray_idx_rgb = torch.randperm(H * W)[:args.pix_rgb // args.deblur_images]
        ret_rgb = self.render(spline_rgb_poses, 
                              ray_idx_rgb.reshape(-1, 1).squeeze(), 
                              H, 
                              W, 
                              K, 
                              args, 
                              sensor_type = "rgb",
                              training=True)

        return ret_event, ret_rgb, ray_idx_event, ray_idx_rgb, events_accu

    def render(self, poses, ray_idx, H, W, K, args, sensor_type, near=0., far=1., training=False):
        if training:
            ray_idx_ = ray_idx.repeat(poses.shape[0])
            poses = poses.unsqueeze(1).repeat(1, ray_idx.shape[0], 1, 1).reshape(-1, 3, 4)
            j = ray_idx_.reshape(-1, 1).squeeze() // W
            i = ray_idx_.reshape(-1, 1).squeeze() % W
            rays_o_, rays_d_ = get_specific_rays(i, j, K, poses)
            rays_o_d = torch.stack([rays_o_, rays_d_], 0)
            batch_rays = torch.permute(rays_o_d, [1, 0, 2])

        else:
            rays_list = []
            for p in poses[:, :3, :4]:
                rays_o_, rays_d_ = get_rays(H, W, K, p)
                rays_o_d = torch.stack([rays_o_, rays_d_], 0)
                rays_list.append(rays_o_d)

            rays = torch.stack(rays_list, 0)
            rays = rays.reshape(-1, 2, H * W, 3)
            rays = torch.permute(rays, [0, 2, 1, 3])

            batch_rays = rays[:, ray_idx]
        batch_rays = batch_rays.reshape(-1, 2, 3)
        batch_rays = torch.transpose(batch_rays, 0, 1)

        # get standard rays
        rays_o, rays_d = batch_rays
        if args.use_viewdirs:
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape
        if args.ndc:
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)

        if args.use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
        viewdirs = rays[:, -3:] if rays.shape[-1] > 8 else None
        bounds = torch.reshape(rays[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]

        t_vals = torch.linspace(0., 1., steps=args.N_samples)
        z_vals = near * (1. - t_vals) + far * (t_vals)
        z_vals = z_vals.expand([N_rays, args.N_samples])

        # perturb
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        raw_output = self.nerf.forward(pts, viewdirs, args)

        rgb_map, disp_map, acc_map, weights, depth_map, sigma = self.nerf.raw2output(self.camera_response_func,
                                                                                     sensor_type,
                                                                                     raw_output, 
                                                                                     z_vals, 
                                                                                     rays_d)

        if args.N_importance > 0:
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], args.N_importance)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

            raw_output = self.nerf_fine.forward(pts, viewdirs, args)
            rgb_map, disp_map, acc_map, weights, depth_map, sigma = self.nerf_fine.raw2output(self.camera_response_func,
                                                                                              sensor_type,
                                                                                              raw_output, 
                                                                                              z_vals,
                                                                                              rays_d)
        ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
        if args.N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['sigma'] = sigma

        return ret
    
    def camera_response_func(self, radience, sensor_type):
        if sensor_type == "rgb":
            color = self.rgb_crf.forward(radience)
            return color 
        elif sensor_type == "event":
            luminance = self.event_crf.forward(radience)
            return luminance
            
    @torch.no_grad()
    def render_video(self, poses, H, W, K, args):
        all_ret = {}
        ray_idx = torch.arange(0, H * W)
        for i in range(0, ray_idx.shape[0], args.chunk):
            ret = self.render(poses, ray_idx[i:i + args.chunk], H, W, K, args)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

        for k in all_ret:
            k_sh = list([H, W]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)
        return all_ret

    @abc.abstractmethod
    def get_pose(self, args, events_ts):
        pass

    @abc.abstractmethod
    def get_pose_rgb(self, args, seg_num=None):
        pass
