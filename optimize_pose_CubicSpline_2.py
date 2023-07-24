import numpy as np
import torch
import torch.nn as nn

import nerf
import spline


class CameraPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Embedding(1, 6)


class TransformPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Embedding(1, 6)


class Model(nerf.Model):
    def __init__(self):
        super().__init__()

    def build_network(self, args):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)

        self.graph.rgb_pose = CameraPose()
        self.graph.transform = TransformPose()

        self.graph.rgb_pose.params.weight.data = torch.nn.Parameter(torch.rand(1, 6) * 0.1)
        self.graph.transform.params.weight.data = torch.nn.Parameter(torch.zeros(1, 6))

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        grad_vars_pose = list(self.graph.rgb_pose.parameters())
        self.optim_pose = torch.optim.Adam(params=grad_vars_pose, lr=args.pose_lrate)

        grad_vars_transform = list(self.graph.transform.parameters())
        self.optim_transform = torch.optim.Adam(params=grad_vars_transform, lr=args.transform_lrate)

        return self.optim, self.optim_pose, self.optim_transform


class Graph(nerf.Graph):
    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True):
        super().__init__(args, D, W, input_ch, input_ch_views, output_ch, skips,
                         use_viewdirs)
        self.pose_eye = torch.eye(3, 4)

    def get_pose(self, i, args, events_ts, poses_ts):
        Dicho_up = poses_ts[:-1][:, np.newaxis].repeat(events_ts.shape[0], axis=1) - events_ts[np.newaxis, :].repeat(
            poses_ts.shape[0] - 1, axis=0)
        Dicho_low = poses_ts[1:][:, np.newaxis].repeat(events_ts.shape[0], axis=1) - events_ts[np.newaxis, :].repeat(
            poses_ts.shape[0] - 1, axis=0)
        judge_func = Dicho_up * Dicho_low

        bound = np.where(judge_func <= 0)
        low_bound = bound[0]
        up_bound = bound[0] + 1

        # start pose
        se3_start = self.transform.params.weight.reshape(1, 1, 6)
        # end pose
        i_0 = torch.tensor((.0, .0, .0, 1.)).reshape(1, 4)
        SE3_from = spline.se3_to_SE3(self.rgb_pose.params.weight.reshape(1, 1, 6)).squeeze()
        SE3_from = torch.cat((SE3_from, i_0), dim=0)
        SE3_trans = spline.se3_to_SE3(self.transform.params.weight.reshape(1, 1, 6)).squeeze()
        SE3_trans = torch.cat((SE3_trans, i_0), dim=0)
        SE3_end = SE3_trans @ SE3_from
        se3_end = spline.SE3_to_se3(SE3_end[:3, :4].reshape(1, 3, 4))

        period = torch.tensor((poses_ts[up_bound] - poses_ts[low_bound])).float()
        t_tau = torch.tensor((events_ts - poses_ts[low_bound])).float()

        spline_poses = spline.SplineEvent(se3_start, se3_end, t_tau, period)

        return spline_poses

    def get_pose_rgb(self, args, seg_num=None):
        # start pose
        se3_start = torch.zeros(1, 1, 6)
        # end pose
        se3_end = self.rgb_pose.params.weight.reshape(1, 1, 6)

        # spline
        if seg_num is None:
            pose_nums = torch.arange(args.deblur_images).reshape(1, -1).repeat(se3_start.shape[0], 1)
            spline_poses = spline.SplineN_linear(se3_start, se3_end, pose_nums, args.deblur_images)
        else:
            pose_nums = torch.arange(seg_num).reshape(1, -1).repeat(se3_start.shape[0], 1)
            spline_poses = spline.SplineN_linear(se3_start, se3_end, pose_nums, seg_num)

        return spline_poses

    def get_pose_render(self):
        spline_poses = spline.se3_to_SE3(self.se3.params.weight)
        return spline_poses

    def get_pose_i(self, pose_i, args, ray_idx):
        ray_idx = ray_idx.reshape([1, -1])
        spline_poses_ = spline.se3_to_SE3(self.se3.params.weight[pose_i])
        spline_poses = spline_poses_.reshape([ray_idx.shape[0], 1, 3, 4]).repeat(1, ray_idx.shape[1], 1, 1).reshape(
            [-1, 3, 4])
        return spline_poses
