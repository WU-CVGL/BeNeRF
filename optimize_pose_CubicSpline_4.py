import torch.nn

import cubicSpline
import nerf


class SE3(torch.nn.Module):
    def __init__(self, shape_1, trajectory_seg_num):
        super().__init__()
        self.start = torch.nn.Embedding(shape_1 // trajectory_seg_num, 6 * 1)  # 22和25
        self.mid = torch.nn.Embedding(shape_1 // trajectory_seg_num, 6 * (trajectory_seg_num - 1))  # 22和25
        self.end = torch.nn.Embedding(shape_1 // trajectory_seg_num * 3 + 1, 6 * 1)  # 22和25


class Model(nerf.Model):
    def __init__(self, se3_start, se3_end):
        super().__init__()
        self.start = se3_start
        self.end = se3_end

    def build_network(self, args):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)
        # self.graph.se3 = torch.nn.Embedding(self.start.shape[0], 6*2)  # 22和25
        # self.graph.se3 = torch.nn.Embedding(self.start.shape[0], 6 * (args.trajectory_seg_num + 1))  # 22和25

        # 初始化
        # torch.nn.init.zeros_(self.graph.se3.weight)
        # low, high = 0.0001, 0.001
        # start = self.end.repeat(1, args.trajectory_seg_num)
        # start = start + (high - low) * torch.rand(self.end.shape[0], 6*args.trajectory_seg_num) + low
        start_end = self.end.repeat(1, 3)
        low, high = 1e-4, 1e-3
        start_end = start_end + (high - low) * torch.rand(start_end.shape[0], start_end.shape[1]) + low
        # self.graph.se3.weight.data = torch.nn.Parameter(start_end)    # 初始 weight 为 [number of images, dims of pose_start + dims of pose_end]

        self.graph.se3 = SE3(self.start.shape[0], args.trajectory_seg_num)

        # self.graph.se3.end.weight.data = torch.nn.Parameter(torch.cat([start_end[:, :6 * 3].reshape(-1,6), start_end[-1:, -6:]], dim=0))
        self.graph.se3.end.weight.data = torch.nn.Parameter(
            torch.cat([start_end[:1, :6 * 3].reshape(-1, 6), start_end[1:, :6], start_end[-1:, -6:]], dim=0))

        if self.graph.se3.end.weight.requires_grad == False:
            print('///*** fix the pose of last line ***///')

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))  # nerf 的 gradient

        grad_vars_se3 = list(self.graph.se3.parameters())
        # self.optim_se3 = torch.optim.SGD(params=grad_vars_se3, lr=args.lrate, momentum=0.8)
        self.optim_se3 = torch.optim.Adam(params=grad_vars_se3, lr=args.lrate)  # se3 的 gradient

        return self.optim, self.optim_se3


class Graph(nerf.Graph):
    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True):
        super().__init__(args, D, W, input_ch, input_ch_views, output_ch, skips,
                         use_viewdirs)  # 继承 nerf 中的 Graph    等价于 nerf.Graph(.......)   继承了父类所有属性，相当于重新构造了一个类
        self.pose_eye = torch.eye(3, 4)

    def get_pose(self, i, img_idx, args, pose_nums, H, trajectory_seg_num):  # pose_nums ：随机选择的 poses 对应的行

        seg_pos_x = torch.arange(pose_nums.shape[0]).reshape([pose_nums.shape[0], 1]).repeat(1, pose_nums.shape[1])
        seg_pos_y = pose_nums // (int(H) // trajectory_seg_num)

        # 段内 H
        H_seg = int(H) // trajectory_seg_num
        # 段内行数 lines number
        pose_nums_seg = pose_nums % H_seg

        select_id = torch.arange(img_idx.shape[0])

        pose0 = self.se3.end.weight[select_id, :6].reshape(pose_nums.shape[0], trajectory_seg_num, 6)[img_idx][
                seg_pos_x, seg_pos_y, :]
        pose1 = self.se3.end.weight[select_id + 1, :6].reshape(pose_nums.shape[0], trajectory_seg_num, 6)[img_idx][
                seg_pos_x, seg_pos_y, :]
        pose2 = self.se3.end.weight[select_id + 2, :6].reshape(pose_nums.shape[0], trajectory_seg_num, 6)[img_idx][
                seg_pos_x, seg_pos_y, :]
        pose3 = self.se3.end.weight[select_id + 3, :6].reshape(pose_nums.shape[0], trajectory_seg_num, 6)[img_idx][
                seg_pos_x, seg_pos_y, :]

        spline_poses = cubicSpline.Spline4N_new(pose0, pose1, pose2, pose3, pose_nums_seg, H_seg, args.delay_time)

        return spline_poses

    def get_pose_new(self, i, img_idx, args, Num):  # pose_nums ：随机选择的 poses 对应的行
        se3_start = self.se3.weight[:, :6][img_idx]  # 使用的 new start and new end 都是 优化后的 se3_weight
        se3_end = self.se3.weight[:, 6:][img_idx]
        pose_nums = torch.arange(Num).reshape(1, -1).repeat(se3_start.shape[0], 1)  # 5 images * Num splined poses
        seg_pos_x = torch.arange(se3_start.shape[0]).reshape([se3_start.shape[0], 1]).repeat(1,
                                                                                             Num)  # 5 images * Num splined poses

        se3_start = se3_start[seg_pos_x, :]
        se3_end = se3_end[seg_pos_x, :]

        spline_poses = cubicSpline.SplineN_new(se3_start, se3_end, pose_nums, Num, delay_time=1)
        return spline_poses

    def get_gt_pose(self, poses, args):
        a = self.pose_eye
        return poses
