import os
import pypose as pp

import numpy as np
import torch

import spline

transdir = os.path.expanduser(r"../logs/trans.npy")

if __name__ == '__main__':
    # Origin c1->w
    R = np.array([0.976781, 0.013642, 0.203538, -0.065466])
    t = np.array([-153.531555, -33.654785, 151.457611]) * 0.01
    origin = pp.SE3(torch.tensor(np.concatenate((t, R[1:], [R[0]]))))

    # Trans c2->c1
    R_t = np.array([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]], dtype=np.float64)
    # x_5cm
    t_t = np.array([0.05, 0., 0.], dtype=np.float64)
    # x_10cm
    # t_t = np.array([0.1, 0., 0.], dtype=np.float64)
    # y_5cm
    # t_t = np.array([0., 0.05, 0.], dtype=np.float64)
    # y_10cm
    # t_t = np.array([0., 0.1, 0.], dtype=np.float64)
    # z_5cm
    # t_t = np.array([0., 0., 0.05], dtype=np.float64)
    # z_10cm
    # t_t = np.array([0., 0., 0.1], dtype=np.float64)
    T = np.concatenate((R_t, t_t.reshape(3, 1)), axis=1)
    trans = pp.mat2SE3(T)

    result = origin @ trans
    print("tx ty tz qx qy qz qw")
    print(result.tensor().numpy())

    # output trans
    np.save(transdir, T)
    print("===== SE3 for trans =====")
    print(T)

