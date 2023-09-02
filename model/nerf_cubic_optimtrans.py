import torch

import spline
from model import nerf
from model.component import CameraPose, EventPose, ExposureTime


class Model(nerf.Model):
    def __init__(self, args, pose_ts):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)
        self.graph.exposure_time = ExposureTime()
        self.graph.exposure_time.params.weight.data = torch.concatenate(
            (torch.nn.Parameter(torch.tensor(pose_ts[0], dtype=torch.float32).reshape((1, 1))),
             torch.nn.Parameter(torch.tensor(pose_ts[1], dtype=torch.float32).reshape((1, 1)))))

    def build_network(self, args, poses=None, event_poses=None):
        self.graph.rgb_pose = CameraPose(4)
        self.graph.event = EventPose(1)

        parm_rgb = torch.concatenate(
            (torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01))
        self.graph.rgb_pose.params.weight.data = torch.nn.Parameter(parm_rgb)

        if args.loadtrans:
            se3_trans = spline.SE3_to_se3(torch.tensor(event_poses[..., :4]))
            parm_e = torch.nn.Parameter(se3_trans)
        else:
            parm_e = torch.nn.Parameter(torch.zeros(1, 6))

        self.graph.event.params.weight.data = torch.nn.Parameter(parm_e)

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
        start, end = self.get_exposure_time()
        period = end - start
        t_tau = events_ts - start

        i_0 = torch.tensor((.0, .0, .0, 1.)).reshape(1, 4)
        SE3_0_from = spline.se3_to_SE3(self.rgb_pose.params.weight[0].reshape(1, 1, 6)).squeeze()
        SE3_0_from = torch.cat((SE3_0_from, i_0), dim=0)
        SE3_1_from = spline.se3_to_SE3(self.rgb_pose.params.weight[1].reshape(1, 1, 6)).squeeze()
        SE3_1_from = torch.cat((SE3_1_from, i_0), dim=0)
        SE3_2_from = spline.se3_to_SE3(self.rgb_pose.params.weight[2].reshape(1, 1, 6)).squeeze()
        SE3_2_from = torch.cat((SE3_2_from, i_0), dim=0)
        SE3_3_from = spline.se3_to_SE3(self.rgb_pose.params.weight[3].reshape(1, 1, 6)).squeeze()
        SE3_3_from = torch.cat((SE3_3_from, i_0), dim=0)

        SE3_trans = spline.se3_to_SE3(self.transform.params.weight.reshape(1, 1, 6)).squeeze()
        SE3_trans = torch.cat((SE3_trans, i_0), dim=0)

        SE3_0 = SE3_0_from @ SE3_trans
        se3_0 = spline.SE3_to_se3(SE3_0[:3, :4].reshape(1, 3, 4))
        SE3_1 = SE3_1_from @ SE3_trans
        se3_1 = spline.SE3_to_se3(SE3_1[:3, :4].reshape(1, 3, 4))
        SE3_2 = SE3_2_from @ SE3_trans
        se3_2 = spline.SE3_to_se3(SE3_2[:3, :4].reshape(1, 3, 4))
        SE3_3 = SE3_3_from @ SE3_trans
        se3_3 = spline.SE3_to_se3(SE3_3[:3, :4].reshape(1, 3, 4))

        spline_poses = spline.spline_event_cubic(se3_0, se3_1, se3_2, se3_3, t_tau, period)

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

    def get_exposure_time(self):
        return self.exposure_time.params.weight[0], self.exposure_time.params.weight[1]
