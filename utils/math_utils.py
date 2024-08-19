import torch
import numpy as np

def safe_log(x, eps=1e-9):
    return torch.log(x + eps)

def lin_log(color, linlog_thres=20):
    color = color * 255
    lin_slope = safe_log(torch.tensor(linlog_thres)) / linlog_thres
    lin_log_rgb = torch.where(color < linlog_thres, lin_slope * color, safe_log(color))
    return lin_log_rgb

log_func = {
    "safelog": safe_log,
    "linlog": lin_log
}

def rgb2brightlog(rgb, dataset_type):
    if dataset_type in ["BeNeRF_Blender", "BeNeRF_Unreal"]:
        brightness_logarithm = log_func["safelog"](rgb)
    elif dataset_type in ["E2NeRF_Synthetic", "E2NeRF_Real"]:
        brightness_logarithm = log_func["linlog"](rgb)
    return brightness_logarithm
