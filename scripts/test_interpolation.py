import torch

import spline

if __name__ == '__main__':
    """
    Test the equality of interpolation before/after T
    """
    node0 = torch.tensor([[0, 0, 0, 0.55, 0.55, 0.55]])
    node1 = torch.tensor([[0.05, 0.03, 0.01, 0.25, 0.59, 0.51]])
    node2 = torch.tensor([[0.06, 0.03, 0.02, 0.23, 0.64, 0.50]])
    node3 = torch.tensor([[0.07, 0.04, 0.03, 0.20, 0.69, 0.47]])

    i_0 = torch.tensor((.0, .0, .0, 1.)).reshape(1, 4)
    T = torch.tensor([0.01, 0.02, 0.01, 0.02, 0.03, -0.02])
    T_SE3 = torch.cat((spline.se3_to_SE3(T), i_0), dim=0)

    pose_nums = torch.arange(9).reshape(1, -1).repeat(node0.shape[0], 1)

    linear_spline = spline.spline_linear(node0, node3, pose_nums, 9)
    cubic_spline = spline.spline_cubic(node0, node1, node2, node3, pose_nums, 9)

    linear_spline = linear_spline @ T_SE3
    cubic_spline = cubic_spline @ T_SE3
    node0 = spline.se3_to_SE3(node0) @ T_SE3
    node1 = spline.se3_to_SE3(node1) @ T_SE3
    node2 = spline.se3_to_SE3(node2) @ T_SE3
    node3 = spline.se3_to_SE3(node3) @ T_SE3
    node0 = torch.unsqueeze(spline.SE3_to_se3(node0[:3, :4].reshape(1, 3, 4)), dim=0)
    node1 = torch.unsqueeze(spline.SE3_to_se3(node1[:3, :4].reshape(1, 3, 4)), dim=0)
    node2 = torch.unsqueeze(spline.SE3_to_se3(node2[:3, :4].reshape(1, 3, 4)), dim=0)
    node3 = torch.unsqueeze(spline.SE3_to_se3(node3[:3, :4].reshape(1, 3, 4)), dim=0)

    linear_spline_ = spline.spline_linear(node0, node3, pose_nums, 9)
    cubic_spline_ = spline.spline_cubic(node0, node1, node2, node3, pose_nums, 9)
    print("")
