import torch.nn

import spline
import nerf_chunk

import numpy as np


class SE3(torch.nn.Module):
    def __init__(self, poses_num, trajectory_seg_num):
        super().__init__()
        self.start = torch.nn.Embedding(poses_num, 6 * 1)  # 22和25
        self.end = torch.nn.Embedding(poses_num, 6 * 1)  # 22和25


class Model(nerf_chunk.Model):
    def __init__(self, poses_se3, poses_ts):
        super().__init__()
        self.poses_se3 = poses_se3
        self.poses_ts = poses_ts

    def build_network(self, args, events):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)

        poses_num = (events['num'] - args.N_window + args.N_window * args.chunk_interval - 1) // (
                    args.N_window * args.chunk_interval)
        self.graph.se3 = SE3(poses_num, args.trajectory_seg_num)

        mid_bound = args.N_window * args.chunk_interval * np.arange(poses_num) + args.N_window // 2
        event_poses_ts = events['ts'][mid_bound]

        Dicho_up = self.poses_ts[:-1][:, np.newaxis].repeat(event_poses_ts.shape[0], axis=1) - event_poses_ts[
                                                                                               np.newaxis, :].repeat(
            self.poses_ts.shape[0] - 1, axis=0)
        Dicho_low = self.poses_ts[1:][:, np.newaxis].repeat(event_poses_ts.shape[0], axis=1) - event_poses_ts[
                                                                                               np.newaxis, :].repeat(
            self.poses_ts.shape[0] - 1, axis=0)

        judge_func = Dicho_up * Dicho_low
        bound = np.where(judge_func <= 0)
        low_bound = bound[0]

        ini_poses_se3_start = self.poses_se3[low_bound]
        low, high = 0.0001, 0.001
        rand = (high - low) * torch.rand(ini_poses_se3_start.shape[0], 6) + low
        ini_poses_se3_end = ini_poses_se3_start + rand

        self.graph.se3.start.weight.data = torch.nn.Parameter(ini_poses_se3_start)
        self.graph.se3.end.weight.data = torch.nn.Parameter(ini_poses_se3_end)

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))  # nerf 的 gradient

        grad_vars_se3 = list(self.graph.se3.parameters())
        self.optim_se3 = torch.optim.Adam(params=grad_vars_se3, lr=args.lrate)  # se3 的 gradient

        return self.optim, self.optim_se3


class Graph(nerf_chunk.Graph):
    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True):
        super().__init__(args, D, W, input_ch, input_ch_views, output_ch, skips,
                         use_viewdirs)  # 继承 nerf 中的 Graph    等价于 nerf.Graph(.......)   继承了父类所有属性，相当于重新构造了一个类
        self.pose_eye = torch.eye(3, 4)

    def get_pose(self, pose_i):  # pose_nums ：随机选择的 poses 对应的行

        poses_se3 = torch.concatenate(
            [self.se3.start.weight[pose_i].reshape([-1, 6]), self.se3.end.weight[pose_i].reshape([-1, 6])])
        spline_poses = spline.se3_to_SE3(poses_se3)

        return spline_poses

    def get_pose_i(self, pose_i, args, ray_idx):  # pose_nums ：随机选择的 poses 对应的行

        ray_idx = ray_idx.reshape([1, -1])
        spline_poses_ = spline.se3_to_SE3(self.se3.end.weight[pose_i])
        spline_poses = spline_poses_.reshape([ray_idx.shape[0], 1, 3, 4]).repeat(1, ray_idx.shape[1], 1, 1).reshape(
            [-1, 3, 4])
        return spline_poses
