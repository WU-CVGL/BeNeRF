import abc

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


class ExposureTime(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Embedding(2, 1)


class Model(nerf.Model):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def build_network(self, args, poses=None, event_poses=None, pose_ts=None, events=None):
        pass

    @abc.abstractmethod
    def setup_optimizer(self, args):
        pass


class Graph(nerf.Graph):
    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True):
        super().__init__(args, D, W, input_ch, input_ch_views, output_ch, skips,
                         use_viewdirs)
        self.pose_eye = torch.eye(3, 4)

    @abc.abstractmethod
    def get_pose(self, args, events_ts, poses_ts):
        pass

    @abc.abstractmethod
    def get_pose_rgb(self, args, seg_num=None):
        pass

    def get_pose_render(self):
        spline_poses = spline.se3_to_SE3(self.se3.params.weight)
        return spline_poses

    def get_pose_i(self, pose_i, args, ray_idx):
        ray_idx = ray_idx.reshape([1, -1])
        spline_poses_ = spline.se3_to_SE3(self.se3.params.weight[pose_i])
        spline_poses = spline_poses_.reshape([ray_idx.shape[0], 1, 3, 4]).repeat(1, ray_idx.shape[1], 1, 1).reshape(
            [-1, 3, 4])
        return spline_poses
