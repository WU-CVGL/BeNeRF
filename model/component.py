import torch
from torch import nn as nn
from utils.mathutils import safelog

class CameraPose(nn.Module):
    def __init__(self, pose_num):
        super(CameraPose, self).__init__()
        self.params = nn.Embedding(pose_num, 6)

class EventPose(nn.Module):
    def __init__(self, pose_num):
        super(EventPose, self).__init__()
        self.params = nn.Embedding(pose_num, 6)

class ExposureTime(nn.Module):
    def __init__(self):
        super(ExposureTime, self).__init__()
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
    
class ColorToneMapper(nn.Module):
    '''A network for color tone-mapping.'''

    def __init__(self, hidden = 0, width = 128, input_type = "Gray"):
        super(ColorToneMapper, self).__init__()
        self.net_hidden = hidden
        self.net_width = width
        self.input_type = str(input_type)

        # RGB
        self.mlp_r = nn.Sequential(*[
            nn.Linear(1, self.net_width),
            nn.ReLU(),
            *[
                nn.Linear(self.net_width, self.net_width), nn.ReLU()
                for i in range(self.net_hidden)
             ],
            nn.Linear(self.net_width, 1)
        ])
        self.mlp_g = nn.Sequential(*[
            nn.Linear(1, self.net_width),
            nn.ReLU(),
            *[
                nn.Linear(self.net_width, self.net_width), nn.ReLU()
                for i in range(self.net_hidden)
             ],
            nn.Linear(self.net_width, 1)
        ])
        self.mlp_b = nn.Sequential(*[
            nn.Linear(1, self.net_width),
            nn.ReLU(),
            *[
                nn.Linear(self.net_width, self.net_width), nn.ReLU()
                for i in range(self.net_hidden)
             ],
            nn.Linear(self.net_width, 1)
        ])

        # Gray
        self.mlp_gray = nn.Sequential(*[
            nn.Linear(1, self.net_width),
            nn.ReLU(),
            *[
                nn.Linear(self.net_width, self.net_width), nn.ReLU()
                for i in range(self.net_hidden)
             ],
            nn.Linear(self.net_width, 1)
        ])

    def forward(self, radience, x, input_exps, noise=None, output_grads=False):
        # logarithmic domain
        log_radience = safelog(radience)

        # Gray
        if self.input_type == "Gray":
            raw_color = self.mlp_gray(log_radience)
            color = torch.tanh(raw_color)

        # RGB
        if self.input_type == "RGB":
            log_r = log_radience[:, 0]
            log_g = log_radience[:, 1]
            log_b = log_radience[:, 2]

            color_r = self.mlp_r(log_r)
            color_g = self.mlp_r(log_g)
            color_b = self.mlp_r(log_b)

            raw_color = torch.cat([color_r, color_g, color_b], -1)
            color = torch.tanh(raw_color)

        return color

class LuminanceToneMapper(nn.Module):
    pass