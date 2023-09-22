import torch

import spline
from model import nerf
from model.component import CameraPose, EventPose


class Model(nerf.Model):
    def __init__(self, args):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)

    def build_network(self, args, poses=None, event_poses=None):
        self.graph.rgb_pose = CameraPose(2)
        self.graph.event = EventPose(1)

        if args.loadpose:
            se3_poses = spline.SE3_to_se3(torch.tensor(poses[..., :4]))
            parm_rgb = torch.nn.Parameter(se3_poses)
        else:
            parm_rgb = torch.nn.Parameter(torch.rand(2, 6) * 0.1)

        self.graph.rgb_pose.params.weight.data = parm_rgb

        parm_e = torch.nn.Parameter(torch.zeros(1, 6))

        self.graph.transform.params.weight.data = parm_e

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate)

        grad_vars_pose = list(self.graph.rgb_pose.parameters())
        self.optim_pose = torch.optim.Adam(params=grad_vars_pose, lr=args.pose_lrate)

        grad_vars_transform = list(self.graph.transform.parameters())
        self.optim_transform = torch.optim.Adam(params=grad_vars_transform, lr=args.transform_lrate)

        return self.optim, self.optim_pose, self.optim_transform


class Graph(nerf.Graph):
    def get_pose(self, args, events_ts):
        i_0 = torch.tensor((.0, .0, .0, 1.)).reshape(1, 4)
        SE3_start_from = spline.se3_to_SE3(self.rgb_pose.params.weight[0].reshape(1, 1, 6)).squeeze()
        SE3_start_from = torch.cat((SE3_start_from, i_0), dim=0)
        SE3_end_from = spline.se3_to_SE3(self.rgb_pose.params.weight[1].reshape(1, 1, 6)).squeeze()
        SE3_end_from = torch.cat((SE3_end_from, i_0), dim=0)

        SE3_trans = spline.se3_to_SE3(self.transform.params.weight.reshape(1, 1, 6)).squeeze()
        SE3_trans = torch.cat((SE3_trans, i_0), dim=0)

        SE3_start = SE3_start_from @ SE3_trans
        se3_start = spline.SE3_to_se3(SE3_start[:3, :4].reshape(1, 3, 4))
        SE3_end = SE3_end_from @ SE3_trans
        se3_end = spline.SE3_to_se3(SE3_end[:3, :4].reshape(1, 3, 4))

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
