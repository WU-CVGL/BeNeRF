import torch
import spline
from model import nerf
from model.component import CameraPose, EventPose
from model.component import ColorToneMapper, LuminanceToneMapper

class Model(nerf.Model):
    def __init__(self, args):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)

    def build_network(self, args, poses=None, event_poses=None):
        self.graph.rgb_pose = CameraPose(4)
        self.graph.transform = EventPose(1)
        self.graph.rgb_crf = ColorToneMapper(hidden = args.rgb_crf_net_hidden, 
                                             width = args.rgb_crf_net_width, 
                                             input_type = "Gray")
        self.graph.event_crf = LuminanceToneMapper(hidden = args.event_crf_net_hidden, 
                                                   width = args.event_crf_net_width, 
                                                   input_type = "Gray")

        parm_rgb = torch.concatenate(
            (torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01))
        self.graph.rgb_pose.params.weight.data = torch.nn.Parameter(parm_rgb)

        # assign zero weights to graph.transform.params
        #parm_e = torch.nn.Parameter(torch.zeros(1, 6))
        parm_e = torch.nn.Parameter(torch.rand(1, 6) * 0.01)
        self.graph.transform.params.weight.data = torch.nn.Parameter(parm_e)

        self.graph.rgb_crf.weights_biases_init()
        self.graph.event_crf.weights_biases_init()        

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params = grad_vars, lr = args.lrate)

        grad_vars_pose = list(self.graph.rgb_pose.parameters())
        self.optim_pose = torch.optim.Adam(params = grad_vars_pose, lr = args.pose_lrate)

        # optim related to trans between cam and event
        grad_vars_transform = list(self.graph.transform.parameters())
        self.optim_transform = torch.optim.Adam(params=grad_vars_transform, lr=args.transform_lrate)

        grad_vars_event_crf = list(self.graph.event_crf.mlp_luminance.parameters())
        self.optim_event_crf = torch.optim.Adam(params = grad_vars_event_crf, lr = args.event_crf_lrate)

        grad_vars_rgb_crf = list(self.graph.rgb_crf.mlp_gray.parameters())
        self.optim_rgb_crf = torch.optim.Adam(params = grad_vars_rgb_crf, lr = args.rgb_crf_lrate)

        return self.optim, self.optim_pose, self.optim_transform, self.optim_rgb_crf, self.optim_event_crf

class Graph(nerf.Graph):
    def get_pose(self, args, events_ts):
        se3_0 = self.rgb_pose.params.weight[0].reshape(1, 1, 6) + self.transform.params.weight.reshape(1, 1, 6)
        se3_1 = self.rgb_pose.params.weight[1].reshape(1, 1, 6) + self.transform.params.weight.reshape(1, 1, 6)
        se3_2 = self.rgb_pose.params.weight[2].reshape(1, 1, 6) + self.transform.params.weight.reshape(1, 1, 6)
        se3_3 = self.rgb_pose.params.weight[3].reshape(1, 1, 6) + self.transform.params.weight.reshape(1, 1, 6)
        #print(self.transform.params.weight.reshape(1, 1, 6))
        # interpolate pose at start and end of time window
        spline_poses = spline.spline_event_cubic(se3_0, se3_1, se3_2, se3_3, events_ts)
        # SE3
        return spline_poses

    def get_pose_rgb(self, args, seg_num=None):
        # init knot pose 
        pose0 = self.rgb_pose.params.weight[0].reshape(1, 1, 6)
        pose1 = self.rgb_pose.params.weight[1].reshape(1, 1, 6)
        pose2 = self.rgb_pose.params.weight[2].reshape(1, 1, 6)
        pose3 = self.rgb_pose.params.weight[3].reshape(1, 1, 6)

        # spline
        if seg_num is None:
            pose_nums = torch.arange(args.deblur_images).reshape(1, -1).repeat(pose0.shape[0], 1)
            spline_poses = spline.spline_cubic(pose0, pose1, pose2, pose3, pose_nums, args.deblur_images)
        else:
            pose_nums = torch.arange(seg_num).reshape(1, -1).repeat(pose0.shape[0], 1)
            spline_poses = spline.spline_cubic(pose0, pose1, pose2, pose3, pose_nums, seg_num)
        # SE3
        return spline_poses