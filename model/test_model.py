import torch
import spline
from model import nerf
from model.component import CameraPose, EventPose
from model.component import ColorToneMapper, LuminanceToneMapper

class Model(nerf.Model):
    def __init__(self, args):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)

    def build_network(self, args, poses=None, event_poses=None):
        self.graph.evt_knot_pose_se3 = CameraPose(4)
        self.graph.rgb_knot_pose_se3 = CameraPose(4)
        self.graph.transform = EventPose(1)
        self.graph.rgb_crf = ColorToneMapper(hidden = args.rgb_crf_net_hidden, 
                                             width = args.rgb_crf_net_width, 
                                             input_type = "Gray")
        self.graph.event_crf = LuminanceToneMapper(hidden = args.event_crf_net_hidden, 
                                                   width = args.event_crf_net_width, 
                                                   input_type = "Gray")

        parm_evt = torch.concatenate(
            (torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01, torch.rand(1, 6) * 0.01))
        self.graph.evt_knot_pose_se3.params.weight.data = torch.nn.Parameter(parm_evt)

        # assign zero weights to graph.transform.params
        #parm_trans = torch.nn.Parameter(torch.zeros(1, 6))
        parm_trans = torch.nn.Parameter(torch.rand(1, 6) * 0.01)
        self.graph.transform.params.weight.data = torch.nn.Parameter(parm_trans)

        self.graph.rgb_crf.weights_biases_init()
        self.graph.event_crf.weights_biases_init()        

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params = grad_vars, lr = args.lrate)

        grad_vars_pose = list(self.graph.evt_knot_pose_se3.parameters())
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
    def get_pose_evt(self, args, events_ts, seg_num = None):
        se3_knot_0 = self.evt_knot_pose_se3.params.weight[0].reshape(1, 1, 6)
        se3_knot_1 = self.evt_knot_pose_se3.params.weight[1].reshape(1, 1, 6)
        se3_knot_2 = self.evt_knot_pose_se3.params.weight[2].reshape(1, 1, 6)
        se3_knot_3 = self.evt_knot_pose_se3.params.weight[3].reshape(1, 1, 6)

        # decide number of interpolation
        pose_nums = None
        if seg_num is None:
            pose_nums = 2
        else:
            pose_nums = seg_num

        events_ts = torch.linspace(events_ts[0], events_ts[1], pose_nums) 

        # interpolate pose at start and end of event time window
        spline_poses = spline.cubic_spline_pose_per_unit_time(
            se3_knot_0, se3_knot_1, se3_knot_2, se3_knot_3, events_ts
        )

        # SE3
        return spline_poses

    def get_pose_rgb(self, args, exposure_ts, seg_num = None):
        # transform in se3 domain
        se3_knot_0 = self.evt_knot_pose_se3.params.weight[0].reshape(1, 1, 6) + self.transform.params.weight.reshape(1, 1, 6)
        se3_knot_1 = self.evt_knot_pose_se3.params.weight[1].reshape(1, 1, 6) + self.transform.params.weight.reshape(1, 1, 6)
        se3_knot_2 = self.evt_knot_pose_se3.params.weight[2].reshape(1, 1, 6) + self.transform.params.weight.reshape(1, 1, 6)
        se3_knot_3 = self.evt_knot_pose_se3.params.weight[3].reshape(1, 1, 6) + self.transform.params.weight.reshape(1, 1, 6)

        # decide number of interpolation
        pose_nums = None
        if seg_num is None:
            pose_nums = args.deblur_images
        else:
            pose_nums = seg_num

        # interpolation time
        exposure_ts = torch.linspace(exposure_ts[0], exposure_ts[1], pose_nums)

        # spline pose accor
        spline_poses = spline.cubic_spline_pose_per_unit_time(
            se3_knot_0, se3_knot_1, se3_knot_2, se3_knot_3, exposure_ts
        )

        # SE3
        return spline_poses