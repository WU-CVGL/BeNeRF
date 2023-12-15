import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
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

        # Create NN
        layers = []
        layers.append(nn.Linear(1, self.net_width))
        layers.append(nn.ReLU())
        for _ in range(self.net_hidden):
            layers.append(nn.Linear(self.net_width, self.net_width))
            layers.append(nn.ReLU())
        layers.append(self.net_width, 1)

        # Gray
        if self.input_type == "Gray":
            self.mlp_gray = nn.Sequential(*layers)
        # RGB
        elif self.input_type == "RGB":
            self.mlp_r = nn.Sequential(*layers)
            self.mlp_g = nn.Sequential(*layers)
            self.mlp_b = nn.Sequential(*layers)
            
    def weights_biases_init(self):
        # Initialize weight and biases
        if self.input_type == "Gray":
            for layer in self.mlp_gray:
                if isinstance(layer, nn.Linear):
                    init.xavier_uniform_(layer.weight)
                    init.zeros_(layer.bias)                   
        elif self.input_type == "RGB":
            for layer_r, layer_g, layer_b in zip(self.mlp_r, self.mlp_g, self.mlp_b):
                layer_list = [layer_r, layer_g, layer_b]
                for layer in layer_list:
                    if isinstance(layer, nn.Linear):
                        init.xavier_uniform_(layer.weight)
                        init.zeros_(layer.bias)        
          
    def forward(self, radience, x, input_exps, noise=None, output_grads=False):
        # logarithmic domain
        log_radience = safelog(radience)

        # tone mapping for camera data
        # Gray
        if self.input_type == "Gray":
            raw_color = self.mlp_gray(log_radience)
            color = F.tanh(raw_color)
        # RGB
        elif self.input_type == "RGB":
            log_radience_r = log_radience[:, 0]
            log_radience_g = log_radience[:, 1]
            log_radience_b = log_radience[:, 2]

            raw_color_r = self.mlp_r(log_radience_r)
            raw_color_g = self.mlp_g(log_radience_g)
            raw_color_b = self.mlp_b(log_radience_b)

            raw_color = torch.cat([raw_color_r, raw_color_g, raw_color_b], -1)
            color = F.tanh(raw_color)

        return color

class LuminanceToneMapper(nn.Module):
    '''A network for luminance tone-mapping.'''

    def __init__(self, hidden = 0, width = 128, input_type = "Gray"):
        super(LuminanceToneMapper, self).__init__()
        self.net_hidden = hidden
        self.net_width = width
        self.input_type = str(input_type)

        # Create NN
        layers = []
        if self.input_type == "Gray":
            layers.append(nn.Linear(1, self.net_width))
        elif self.input_type == "RGB":
            layers.append(nn.Linear(3, self.net_width))
        layers.append(nn.ReLU())
        for _ in range(self.net_hidden):
            layers.append(nn.Linear(self.net_width, self.net_width))
            layers.append(nn.ReLU())
        layers.append(self.net_width, 1)

        # Luminance
        self.mlp_luminance = nn.Sequential(*layers)

    def weights_biases_init(self):
        for layer in self.mlp_luminance:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, radience, x, input_exps, noise=None, output_grads=False):
        # logarithmic domain
        log_radience = safelog(radience)

        # tone mapping for event data
        raw_luminance = self.mlp_luminance(log_radience)
        luminance = F.relu(raw_luminance)

        return luminance
