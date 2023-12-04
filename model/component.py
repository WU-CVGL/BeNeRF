from torch import nn as nn


class CameraPose(nn.Module):
    def __init__(self, pose_num):
        super(CameraPose).__init__()
        self.params = nn.Embedding(pose_num, 6)


class EventPose(nn.Module):
    def __init__(self, pose_num):
        super(EventPose).__init__()
        self.params = nn.Embedding(pose_num, 6)


class ExposureTime(nn.Module):
    def __init__(self):
        super(ExposureTime).__init__()
        self.params = nn.Embedding(2, 1)


class CRF(nn.Module):
    def __init__(self, in_rgbs, hidden, *args, **kwargs) -> None:
        super(CRF).__init__(*args, **kwargs)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_rgbs, hidden),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden, 1),
        )


    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        return x