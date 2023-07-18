import torch.nn

import spline
import test_nerf

import numpy as np


class SE3(torch.nn.Module):
    def __init__(self, shape_1, trajectory_seg_num):
        super().__init__()
        self.start = torch.nn.Embedding(shape_1 // trajectory_seg_num, 6 * 1)
        self.end = torch.nn.Embedding(shape_1 // trajectory_seg_num, 6 * 1)
        self.mid = torch.nn.Embedding(shape_1 // trajectory_seg_num, 6 * (trajectory_seg_num - 1))


class Model(test_nerf.Model):
    def __init__(self, poses_se3, poses_ts):
        super().__init__()
        self.poses_se3 = poses_se3
        self.poses_ts = poses_ts

    def build_network(self, args):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)

        self.graph.se3 = SE3(self.poses_se3.shape[0], args.trajectory_seg_num)

        self.graph.se3.end.weight.data = torch.nn.Parameter(self.poses_se3)

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        grad_vars_se3 = list(self.graph.se3.parameters())
        self.optim_se3 = torch.optim.Adam(params=grad_vars_se3, lr=args.lrate)

        return self.optim, self.optim_se3


class Graph(test_nerf.Graph):
    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True):
        super().__init__(args, D, W, input_ch, input_ch_views, output_ch, skips,
                         use_viewdirs)
        self.pose_eye = torch.eye(3, 4)

    def get_pose(self, pose_i):

        poses_se3 = torch.concatenate(
            [self.se3.end.weight[pose_i].reshape([-1, 6]), self.se3.end.weight[pose_i + 1].reshape([-1, 6])])
        spline_poses = spline.se3_to_SE3(poses_se3)

        return spline_poses

    def get_pose_i(self, pose_i, args, ray_idx):

        ray_idx = ray_idx.reshape([1, -1])
        spline_poses_ = spline.se3_to_SE3(self.se3.end.weight[pose_i])
        spline_poses = spline_poses_.reshape([ray_idx.shape[0], 1, 3, 4]).repeat(1, ray_idx.shape[1], 1, 1).reshape(
            [-1, 3, 4])
        return spline_poses
