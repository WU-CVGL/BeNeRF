import os
import pypose as pp

import numpy as np
import torch

import spline

transdir = os.path.expanduser(r"../logs/trans.npy")

if __name__ == '__main__':
    # Living room Origin c1->w
    # R = np.array([0.976781, 0.013642, 0.203538, -0.065466])
    # t = np.array([-153.531555, -33.654785, 151.457611]) * 0.01

    # White room Origin c1->w
    # R = np.array([0.957979, 0.040439, 0.226936, -0.170707])
    # t = np.array([-54.199661, -22.093983, 150.290070]) * 0.01
    # origin = pp.SE3(torch.tensor(np.concatenate((t, R[1:], [R[0]]))))

    # pinkcastle   Origin c1->w
    # R = np.array([0.999356, 0.000100, -0.035774, 0.002800])
    # t = np.array([4900.291504, 0.000000, 1622.000732]) * 0.01
    # origin = pp.SE3(torch.tensor(np.concatenate((t, R[1:], [R[0]]))))

    # tanabata   Origin c1->w
    # R = np.array([0.6097474694252014, 0.42423614859580994, 0.382367879152298, 0.5495694279670715])
    # t = np.array([15.299233436584473, -3.744455337524414, 11.704211235046387])
    # origin = pp.SE3(torch.tensor(np.concatenate((t, R[1:], [R[0]]))))

    # ourdoorpool  Origin c1->w
    R = np.array([0.449483186006546, 0.3870837092399597, 0.4603446424007416, 0.660464882850647])
    t = np.array([16.65660285949707, 3.878607749938965, 3.990011215209961])
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

