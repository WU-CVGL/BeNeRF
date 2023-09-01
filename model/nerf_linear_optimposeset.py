import torch

from model import nerf
import spline
from model.component import CameraPose, ExposureTime


class Model(nerf.Model):
    def __init__(self, args, pose_ts):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)
        self.graph.exposure_time = ExposureTime()
        self.graph.exposure_time.params.weight.data = torch.concatenate(
            (torch.nn.Parameter(torch.tensor(pose_ts[0], dtype=torch.float32).reshape((1, 1))),
             torch.nn.Parameter(torch.tensor(pose_ts[1], dtype=torch.float32).reshape((1, 1)))))

    def build_network(self, args, poses=None, event_poses=None):
        self.graph.rgb_pose = CameraPose(2)
        self.graph.rgb_pose.params.weight.data = torch.nn.Parameter(torch.rand(2, 6) * 0.1)
        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate)

        grad_vars_pose = list(self.graph.rgb_pose.parameters())
        self.optim_pose = torch.optim.Adam(params=grad_vars_pose, lr=args.pose_lrate)

        # Fake optimizer
        self.optim_transform = torch.optim.Adam(params=[torch.nn.Parameter(torch.tensor(.0))], lr=args.transform_lrate)

        return self.optim, self.optim_pose, self.optim_transform


class Graph(nerf.Graph):
    def get_pose(self, args, events_ts):
        start = self.exposure_time.params.weight[0]
        end = self.exposure_time.params.weight[1]
        period = end - start
        t_tau = events_ts - start

        se3_start = self.rgb_pose.params.weight[0].reshape(1, 1, 6)
        se3_end = self.rgb_pose.params.weight[1].reshape(1, 1, 6)

        spline_poses = spline.spline_event_linear(se3_start, se3_end, t_tau, period)

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
