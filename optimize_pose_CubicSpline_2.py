import torch.nn

import cubicSpline
import nerf

import numpy as np


class SE3(torch.nn.Module):
    def __init__(self, shape_1, trajectory_seg_num):
        super().__init__()
        self.params = torch.nn.Embedding(shape_1, 6)  # 22和25
        # self.rgb_params = torch.nn.Embedding(2, 6)

class Model(nerf.Model):
    def __init__(self, poses_se3, poses_ts):
        super().__init__()
        self.poses_se3 = poses_se3
        self.poses_ts = poses_ts

    def build_network(self, args):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)

        self.graph.se3 = SE3(self.poses_se3.shape[0], args.trajectory_seg_num)

        self.graph.se3.params.weight.data = torch.nn.Parameter(self.poses_se3)
        # self.graph.se3.rgb_params.weight.data = torch.nn.Parameter(self.poses_se3[:2])

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))  # nerf 的 gradient

        grad_vars_se3 = list(self.graph.se3.parameters())
        self.optim_se3 = torch.optim.Adam(params=grad_vars_se3, lr=args.lrate)  # se3 的 gradient

        return self.optim, self.optim_se3


class Graph(nerf.Graph):
    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True):
        super().__init__(args, D, W, input_ch, input_ch_views, output_ch, skips,
                         use_viewdirs)  # 继承 nerf 中的 Graph    等价于 nerf.Graph(.......)   继承了父类所有属性，相当于重新构造了一个类
        self.pose_eye = torch.eye(3, 4)

    def get_pose(self, i, args, events_ts, poses_ts, trajectory_seg_num):  # pose_nums ：随机选择的 poses 对应的行

        Dicho_up = poses_ts[:-1][:, np.newaxis].repeat(events_ts.shape[0], axis=1) - events_ts[np.newaxis, :].repeat(
            poses_ts.shape[0] - 1, axis=0)
        Dicho_low = poses_ts[1:][:, np.newaxis].repeat(events_ts.shape[0], axis=1) - events_ts[np.newaxis, :].repeat(
            poses_ts.shape[0] - 1, axis=0)
        judge_func = Dicho_up * Dicho_low

        bound = np.where(judge_func <= 0)
        low_bound = bound[0]
        up_bound = bound[0] + 1

        se3_start = self.se3.params.weight[low_bound, :]
        se3_end = self.se3.params.weight[up_bound, :]

        period = torch.tensor((poses_ts[up_bound] - poses_ts[low_bound])).float()
        t_tau = torch.tensor((events_ts - poses_ts[low_bound])).float()

        spline_poses = cubicSpline.SplineEvent(se3_start, se3_end, t_tau, period, args.delay_time)

        return spline_poses

    def get_pose_rgb(self, args):
        se3_start = self.se3.params.weight[0, :].reshape(1, 1, 6)
        se3_end = self.se3.params.weight[1, :].reshape(1, 1, 6)
        pose_nums = torch.arange(args.deblur_images).reshape(1, -1).repeat(se3_start.shape[0], 1)

        spline_poses = cubicSpline.SplineN_linear(se3_start, se3_end, pose_nums, args.deblur_images)
        return spline_poses

    def get_pose_render(self):
        spline_poses = cubicSpline.se3_to_SE3(self.se3.params.weight)
        return spline_poses

    def get_pose_i(self, pose_i, args, ray_idx):  # pose_nums ：随机选择的 poses 对应的行
        ray_idx = ray_idx.reshape([1, -1])
        spline_poses_ = cubicSpline.se3_to_SE3(self.se3.params.weight[pose_i])
        spline_poses = spline_poses_.reshape([ray_idx.shape[0], 1, 3, 4]).repeat(1, ray_idx.shape[1], 1, 1).reshape(
            [-1, 3, 4])
        return spline_poses
