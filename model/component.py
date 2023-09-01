from torch import nn as nn


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
