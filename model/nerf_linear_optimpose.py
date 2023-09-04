import torch

import spline
from model import nerf
from model.component import CameraPose, EventPose


class Model(nerf.Model):
    def __init__(self, args):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)

    def build_network(self, args, poses=None, event_poses=None):
        self.graph.rgb_pose = CameraPose(2)
        self.graph.event = EventPose(2)

        if args.loadpose:
            se3_poses = spline.SE3_to_se3(torch.tensor(poses[..., :4]))
            self.graph.rgb_pose.params.weight.data = torch.nn.Parameter(se3_poses)
            se3_trans = spline.SE3_to_se3(torch.tensor(event_poses[..., :4]))
            self.graph.event.params.weight.data = torch.nn.Parameter(se3_trans)
        else:
            self.graph.rgb_pose.params.weight.data = torch.nn.Parameter(torch.rand(2, 6) * 0.1)
            self.graph.event.params.weight.data = torch.nn.Parameter(torch.rand(2, 6) * 0.1)

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
    def get_pose(self, args, events_ts):
        se3_start = self.event.params.weight[0].reshape(1, 1, 6)
        se3_end = self.event.params.weight[1].reshape(1, 1, 6)

        spline_poses = spline.spline_event_linear(se3_start, se3_end, events_ts)

        return spline_poses

    def get_pose_rgb(self, args, seg_num=None):
        se3_start = self.rgb_pose.params.weight[0].reshape(1, 1, 6)
        se3_end = self.rgb_pose.params.weight[1].reshape(1, 1, 6)

        # spline
        if seg_num is None:
            pose_nums = torch.arange(args.deblur_images).reshape(1, -1).repeat(se3_start.shape[0], 1)
            spline_poses = spline.spline_linear(se3_start, se3_end, pose_nums, args.deblur_images)
        else:
            pose_nums = torch.arange(seg_num).reshape(1, -1).repeat(se3_start.shape[0], 1)
            spline_poses = spline.spline_linear(se3_start, se3_end, pose_nums, seg_num)

        return spline_poses
