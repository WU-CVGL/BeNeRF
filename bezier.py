import torch
import spline
import scipy.special
import numpy as np
import torch.nn as nn

def compute_bezier_coefficient_mat(sample_time, order_curve):
    # Uniformly sample n points on t: [0, 1].
    t = sample_time
    
    binom_coeff = [scipy.special.binom(order_curve, k) for k in range(order_curve+1)]
    # Build coefficient matrix.
    bezier_coeff = []
    for i in range(order_curve+1):
        coeff_i = binom_coeff[i] * torch.pow(1-t, order_curve-i) * torch.pow(t, i)
        bezier_coeff.append(coeff_i)

    bezier_coeff = torch.stack(bezier_coeff, dim=-1)

    return bezier_coeff

def cubic_bezier_poses_unit_time(knot_0, knot_1, knot_2, knot_3, sample_time):
    # avoid numerial computation issues
    pos_0 = torch.where(sample_time == 0)
    sample_time[pos_0] = sample_time[pos_0] + 0.000001
    pos_1 = torch.where(sample_time == 1)
    sample_time[pos_1] = sample_time[pos_1] - 0.000001
    sample_time = sample_time.unsqueeze(-1)

    # cubic bezier
    order = 3
    bezier_coeff = compute_bezier_coefficient_mat(sample_time, order)
    
    # se3 to q&t
    q0, t0 = spline.se3_2_qt_parallel(knot_0)
    q1, t1 = spline.se3_2_qt_parallel(knot_1)
    q2, t2 = spline.se3_2_qt_parallel(knot_2)
    q3, t3 = spline.se3_2_qt_parallel(knot_3)

    # interpolate pose
    t0 = t0.reshape(1,3)
    t1 = t1.reshape(1,3)
    t2 = t2.reshape(1,3)
    t3 = t3.reshape(1,3)
    t = torch.cat((t0, t1, t2, t3), dim = 0)

    interpolated_poses = torch.matmul(bezier_coeff, t)

    q_01 = spline.q_to_Q_parallel(spline.q_to_q_conj_parallel(q0)) @ q1[..., None]  # [1]
    q_12 = spline.q_to_Q_parallel(spline.q_to_q_conj_parallel(q1)) @ q2[..., None]  # [2]
    q_23 = spline.q_to_Q_parallel(spline.q_to_q_conj_parallel(q2)) @ q3[..., None]  # [3]

    r_01 = spline.log_q2r_parallel(q_01.squeeze(-1)).reshape(1,3)  # [4]
    r_12 = spline.log_q2r_parallel(q_12.squeeze(-1)).reshape(1,3) # [5]
    r_23 = spline.log_q2r_parallel(q_23.squeeze(-1)).reshape(1,3)  # [6]
    interpolated_r_01 = torch.matmul(bezier_coeff[:,1], r_01)
    interpolated_r_12 = torch.matmul(bezier_coeff[:,1], r_12)
    interpolated_r_23 = torch.matmul(bezier_coeff[:,1], r_23)

    q_t_0 = spline.exp_r2q_parallel(interpolated_r_01)  # [7]
    q_t_1 = spline.exp_r2q_parallel(interpolated_r_12)  # [8]
    q_t_2 = spline.exp_r2q_parallel(interpolated_r_23)  # [9]

    q_product1 = spline.q_to_Q_parallel(q_t_1) @ q_t_2[..., None]  # [10]
    q_product2 = spline.q_to_Q_parallel(q_t_0) @ q_product1  # [10]
    q_t = spline.q_to_Q_parallel(q0) @ q_product2  # [10]

    R = spline.q_to_R_parallel(q_t.squeeze(-1))  # [3,3]
    t = t.unsqueeze(dim=-1)

    pose_spline = torch.cat([R, t], -1)  # [3, 4]
    poses = pose_spline.reshape([-1, 3, 4])  # [35, 6, 3, 4]

    return poses