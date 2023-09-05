import numpy as np
import torch

import spline
from model import nerf
from model.component import CameraPose, ExposureTime
from utils.eventutils import accumulate_events_range


class Model(nerf.Model):
    def __init__(self, args):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)
        self.graph.exposure_time = ExposureTime()
        self.graph.exposure_time.params.weight.data = torch.concatenate(
            (torch.nn.Parameter(torch.tensor(.0, dtype=torch.float32).reshape((1, 1))),
             torch.nn.Parameter(torch.tensor(.0, dtype=torch.float32).reshape((1, 1)))))

    def build_network(self, args, poses=None, event_poses=None):
        self.graph.rgb_pose = CameraPose(4)
        parm_rgb = torch.concatenate(
            (torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01))
        self.graph.rgb_pose.params.weight.data = torch.nn.Parameter(parm_rgb)

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate)

        grad_vars_pose = list(self.graph.rgb_pose.parameters())
        self.optim_pose = torch.optim.Adam(params=grad_vars_pose, lr=args.pose_lrate)

        # exposure_time optimizer
        grad_vars_exposure = list(self.graph.exposure_time.parameters())
        self.optim_transform = torch.optim.Adam(params=grad_vars_exposure, lr=args.transform_lrate)

        return self.optim, self.optim_pose, self.optim_transform


class Graph(nerf.Graph):

    def get_pose(self, args, events_ts):
        pose0 = self.rgb_pose.params.weight[0].reshape(1, 1, 6)
        pose1 = self.rgb_pose.params.weight[1].reshape(1, 1, 6)
        pose2 = self.rgb_pose.params.weight[2].reshape(1, 1, 6)
        pose3 = self.rgb_pose.params.weight[3].reshape(1, 1, 6)

        spline_poses = spline.spline_event_cubic(pose0, pose1, pose2, pose3, events_ts)

        return spline_poses

    def get_pose_rgb(self, args, seg_num=None):
        pose0 = self.rgb_pose.params.weight[0].reshape(1, 1, 6)
        pose1 = self.rgb_pose.params.weight[1].reshape(1, 1, 6)
        pose2 = self.rgb_pose.params.weight[2].reshape(1, 1, 6)
        pose3 = self.rgb_pose.params.weight[3].reshape(1, 1, 6)

        # spline
        if seg_num is None:
            pose_nums = torch.arange(args.deblur_images).reshape(1, -1).repeat(pose0.shape[0], 1)
            spline_poses = spline.spline_cubic(pose0, pose1, pose2, pose3, pose_nums, args.deblur_images)
        else:
            pose_nums = torch.arange(seg_num).reshape(1, -1).repeat(pose0.shape[0], 1)
            spline_poses = spline.spline_cubic(pose0, pose1, pose2, pose3, pose_nums, seg_num)

        return spline_poses

    def forward(self, i, events, H, W, K, K_event, args):
        if args.time_window:
            start, end = .1, .0
            delta_t = end - start
            window_t = delta_t * args.window_percent
            low_t = torch.rand(1) * (1 - args.window_percent) * delta_t + start
            upper_t = low_t + window_t
            idx_a = low_t <= events["ts"]
            idx_b = events["ts"] <= upper_t
            idx = idx_a * idx_b
            indices = np.where(idx)
            pol_window = events['pol'][indices]
            x_window = events['x'][indices]
            y_window = events['y'][indices]
            ts_window = events['ts'][indices]
        else:
            num = len(events["pol"])
            if args.window_desc:
                # linear desc
                i_end = args.max_iter * args.window_desc_end
                percent = args.window_percent - (args.window_percent - args.window_percent_end) * (
                        i / i_end) if i < i_end else args.window_percent_end
                N_window = round(num * percent)
            else:
                N_window = round(num * args.window_percent)

            if args.random_window:
                window_low_bound = np.random.randint(num - N_window)
            else:
                window_low_bound = np.random.randint((num - N_window) // N_window) * N_window

            window_up_bound = int(window_low_bound + N_window)
            pol_window = events['pol'][window_low_bound:window_up_bound]
            x_window = events['x'][window_low_bound:window_up_bound]
            y_window = events['y'][window_low_bound:window_up_bound]
            ts_window = events['ts'][window_low_bound:window_up_bound]

        out = np.zeros((args.h_event, args.w_event))
        accumulate_events_range(out, x_window, y_window, pol_window)
        out *= args.threshold
        events_accu = torch.tensor(out)

        # timestamps of event windows begin and end
        if args.time_window:
            events_ts = np.stack((low_t, upper_t)).reshape(2)
        else:
            events_ts = ts_window[np.array([0, int(N_window) - 1])]

        ray_idx_event = torch.randperm(args.h_event * args.w_event)[:args.pix_event]
        spline_poses = self.get_pose(args, torch.tensor(events_ts, dtype=torch.float32))
        spline_rgb_poses = self.get_pose_rgb(args)

        # render event
        ret_event = self.render(spline_poses, ray_idx_event.reshape(-1, 1).squeeze(), args.h_event, args.w_event,
                                K_event,
                                args,
                                training=True)

        # render rgb
        ray_idx_rgb = torch.randperm(H * W)[:args.pix_rgb // args.deblur_images]
        ret_rgb = self.render(spline_rgb_poses, ray_idx_rgb.reshape(-1, 1).squeeze(), H, W, K, args,
                              training=True)

        return ret_event, ret_rgb, ray_idx_event, ray_idx_rgb, events_accu
