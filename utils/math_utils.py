import torch
import numpy as np

# def safelog(x, eps=1e-12):
#     x[..., 0:1]
#     dim = x.size()
#     eps = torch.Tensor([eps]).expand(dim)
#     x = torch.max(x, torch.Tensor(eps))
#     return torch.log(x)

def safelog(x, eps=1e-3):
    return torch.log(x + eps)

def lin_log(color, linlog_thres=20):
    """
    Input: 
    :color torch.Tensor of (N_rand_events, 1 or 3). 1 if use_luma, else 3 (rgb).
           We pass rgb here, if we want to treat r,g,b separately in the loss (each pixel must obey event constraint).
    """
    # Convert [0,1] to [0,255]
    color = color * 255
    # Compute the required slope for linear region (below luma_thres)
    # we need natural log (v2e writes ln and "it comes from exponential relation")
    lin_slope = np.log(linlog_thres) / linlog_thres

    # Peform linear-map for smaller thres, and log-mapping for above thresh
    lin_log_rgb = torch.where(color < linlog_thres, lin_slope * color, torch.log(color))
    return lin_log_rgb
