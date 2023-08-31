import numpy as np
import torch
import torch.nn as nn

from model import nerf
import spline


class CameraPose(nn.Module):
    def __init__(self, pose_num):
        super().__init__()
        self.params = nn.Embedding(pose_num, 6)


class EventPose(nn.Module):
    def __init__(self, pose_num):
        super().__init__()
        self.params = nn.Embedding(pose_num, 6)


class Model(nerf.Model):
    def __init__(self):
        super().__init__()

    def build_network(self, args, poses=None, event_poses=None):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)

        if args.cubic_spline:
            self.graph.rgb_pose = CameraPose(4)
            self.graph.event = EventPose(4)
            # se3_poses = spline.SE3_to_se3(torch.tensor(poses[..., :4]))
            # parm_rgb = torch.concatenate(
            #     (se3_poses[0].reshape(1, 6), se3_poses[0].reshape(1, 6) + torch.rand(1) * 0.01,
            #      se3_poses[1].reshape(1, 6), se3_poses[1].reshape(1, 6) + torch.rand(1) * 0.01))
            parm_rgb = torch.concatenate(
                (torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01))
            self.graph.rgb_pose.params.weight.data = torch.nn.Parameter(parm_rgb)

            # se3_trans = spline.SE3_to_se3(torch.tensor(event_poses[..., :4]))
            # parm_e = torch.concatenate(
            #     (se3_trans[0].reshape(1, 6), se3_trans[0].reshape(1, 6) + torch.rand(1) * 0.01,
            #      se3_trans[1].reshape(1, 6), se3_trans[1].reshape(1, 6) + torch.rand(1) * 0.01))
            parm_e = torch.concatenate(
                (torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01))
            self.graph.event.params.weight.data = torch.nn.Parameter(parm_e)

            return self.graph

        # Init for RGB camera
        if args.fix_pose:
            # If pose cannot be optimized
            # start: se3_start (from .npy)
            # end: se3_end (from .npy)
            self.graph.rgb_pose = CameraPose(2)
        else:
            # If pose can be optimized
            # start: Identity
            # end: rgb_pose
            self.graph.rgb_pose = CameraPose(1)

        # Init for event camera
        if args.fix_event_pose:
            # If event cameras are fixed on start and end (from poses_bounds_events.npy)
            self.graph.event = EventPose(2)
        else:
            # If event cameras can be optimized or load from a fix transform (trans.npy)
            self.graph.event = EventPose(1)

        # Parameter for RGB camera
        if args.fix_pose:
            se3_poses = spline.SE3_to_se3(torch.tensor(poses[..., :4]))
            self.graph.rgb_pose.params.weight.data = torch.nn.Parameter(se3_poses)
        else:
            self.graph.rgb_pose.params.weight.data = torch.nn.Parameter(torch.rand(1, 6) * 0.1)

        # Parameter for event camera
        if args.fix_event_pose:
            se3_trans = spline.SE3_to_se3(torch.tensor(event_poses[..., :4]))
            self.graph.event.params.weight.data = torch.nn.Parameter(se3_trans)
        elif args.fix_trans:
            self.graph.event.params.weight.data = torch.nn.Parameter(
                torch.tensor(event_poses.reshape(1, 6).astype(np.float32)))
        else:
            self.graph.event.params.weight.data = torch.nn.Parameter(torch.zeros(1, 6))

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate)

        grad_vars_pose = list(self.graph.rgb_pose.parameters())
        self.optim_pose = torch.optim.Adam(params=grad_vars_pose, lr=args.pose_lrate)

        grad_vars_transform = list(self.graph.event.parameters())
        self.optim_transform = torch.optim.Adam(params=grad_vars_transform, lr=args.transform_lrate)

        return self.optim, self.optim_pose, self.optim_transform


class Graph(nerf.Graph):
    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True):
        super().__init__(args, D, W, input_ch, input_ch_views, output_ch, skips,
                         use_viewdirs)
        self.pose_eye = torch.eye(3, 4)

    def get_pose(self, args, events_ts, poses_ts):
        period = torch.tensor((poses_ts[1] - poses_ts[0])).float()
        t_tau = torch.tensor((events_ts - poses_ts[0])).float()

        if args.cubic_spline:
            pose0 = self.event.params.weight[0].reshape(1, 1, 6)
            pose1 = self.event.params.weight[1].reshape(1, 1, 6)
            pose2 = self.event.params.weight[2].reshape(1, 1, 6)
            pose3 = self.event.params.weight[3].reshape(1, 1, 6)

            spline_poses = spline.spline_event_cubic(pose0, pose1, pose2, pose3, t_tau, period)

            return spline_poses

        if args.fix_event_pose:
            # If event poses are fixed,
            # load from self.event
            se3_start = self.event.params.weight[0].reshape(1, 1, 6)
            se3_end = self.event.params.weight[1].reshape(1, 1, 6)
        elif not args.optimize_event:
            # If do not optimize trans, meaning rgb and event are overlapped
            se3_start = self.rgb_pose.params.weight[0].reshape(1, 1, 6)
            se3_end = self.rgb_pose.params.weight[1].reshape(1, 1, 6)
        else:
            # Use a trans between rgb and events
            # start pose
            se3_start = self.transform.params.weight.reshape(1, 1, 6)
            # end pose
            i_0 = torch.tensor((.0, .0, .0, 1.)).reshape(1, 4)
            SE3_from = spline.se3_to_SE3(self.rgb_pose.params.weight.reshape(1, 1, 6)).squeeze()
            SE3_from = torch.cat((SE3_from, i_0), dim=0)
            SE3_trans = spline.se3_to_SE3(self.transform.params.weight.reshape(1, 1, 6)).squeeze()
            SE3_trans = torch.cat((SE3_trans, i_0), dim=0)
            SE3_end = SE3_from @ SE3_trans
            se3_end = spline.SE3_to_se3(SE3_end[:3, :4].reshape(1, 3, 4))

        spline_poses = spline.spline_event_linear(se3_start, se3_end, t_tau, period)

        return spline_poses

    def get_pose_rgb(self, args, seg_num=None):
        if args.cubic_spline:
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

        if args.fix_pose:
            se3_start = self.rgb_pose.params.weight[0].reshape(1, 1, 6)
            se3_end = self.rgb_pose.params.weight[1].reshape(1, 1, 6)
        else:
            # start pose
            se3_start = torch.zeros(1, 1, 6)
            # end pose
            se3_end = self.rgb_pose.params.weight.reshape(1, 1, 6)

        # spline
        if seg_num is None:
            pose_nums = torch.arange(args.deblur_images).reshape(1, -1).repeat(se3_start.shape[0], 1)
            spline_poses = spline.spline_linear(se3_start, se3_end, pose_nums, args.deblur_images)
        else:
            pose_nums = torch.arange(seg_num).reshape(1, -1).repeat(se3_start.shape[0], 1)
            spline_poses = spline.spline_linear(se3_start, se3_end, pose_nums, seg_num)

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
