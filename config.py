import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    ## config 
    parser.add_argument("--device", type=int, default=0,
                        help='cuda id to use')
    parser.add_argument("--debug", action='store_true',
                        help='whether to use random seed')
    parser.add_argument("--seed", type=int, default=0,
                        help='which seed to use')
    parser.add_argument('--config', is_config_file=True, default='./configs/e2nerf/real/camera/0.txt',
                        help='config file path')
    parser.add_argument("--project", type=str, default="None",
                        help='the project name')
    parser.add_argument("--expname", type = str, 
                        help = 'the experiment name')
    parser.add_argument("--datadir", type = str, 
                        help = 'input data directory')
    parser.add_argument("--logdir", type = str, 
                        help = 'logs directory')
    parser.add_argument("--dataset", type = str, 
                        help = 'the dataset name')
    parser.add_argument("--index", type=int, default=0,
                        help='the index of the image in the dataset to deblur')
    
    ## viewer
    parser.add_argument("--viewer", type=str, default="wandb",
                        help='the viewer to use')
    parser.add_argument("--depth", action='store_true',
                        help='whether to view depth rendering results')

    ## model options
    parser.add_argument("--model", type=str, default='benerf',
                        help='model type to use')
    parser.add_argument("--load_checkpoint", action='store_true',
                        help='whethet to use model checkpoint already exists')
    parser.add_argument("--loadpose", action='store_true',
                        help='the viewer to use (wandb)')
    parser.add_argument("--loadtrans", action='store_true',
                        help='the viewer to use (wandb)')
    parser.add_argument("--traj", type=str, default='spline',
                        help='representation for camera trajectory')
    parser.add_argument("--num_interpolated_pose", type=int, default=19,
                        help='the number of poses interpolated from spline trajectory')
    parser.add_argument("--use_barf_c2f", action='store_true',
                        help = "whether to use barf strategy to optimize pose")
    parser.add_argument("--barf_c2f_start", type=float, default=0.1,
                        help='iteration step when starts coarse to fine pose optimization')
    parser.add_argument("--barf_c2f_end", type=float, default=0.5,
                        help='iteration step when ends coarse to fine pose optimization')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--rgb_crf_net_hidden", type = int, default = 0,
                        help = "the number of hidden layer in rgb_crf")
    parser.add_argument("--rgb_crf_net_width", type = int, default = 128,
                        help = "the width of linear layer in rgb_crf")
    parser.add_argument("--event_crf_net_hidden", type = int, default = 0,
                        help = "the number of hidden layer in event_crf")
    parser.add_argument("--event_crf_net_width", type = int, default = 128,
                        help = "the width of linear layer in event_crf") 
    parser.add_argument("--chunk", type=int, default=1024 * 4,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 32,
                        help='number of pts sent through network in parallel, decrease if running out of memory')  
    parser.add_argument("--channels", type=int, default=3,
                        help='whether to use 3-channel or single-channel images')
    parser.add_argument("--sampling_event_rays", type=int, default=2048,
                        help='number of sampled rays for Event camera')
    parser.add_argument("--sampling_rgb_rays", type=int, default=1024,
                        help='number of sampled rays for RGB camera')
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    
    ## render test
    parser.add_argument("--render_images", action='store_true',
                        help='whether to render images when testing')
    parser.add_argument("--render_video", action='store_true',
                        help='whether to render video when testing')
    parser.add_argument("--extract_poses", action='store_true',
                        help='whether to extract poses when testing')
    parser.add_argument("--checkpoint", type=int, default=80000,
                        help='which checkpoint to load')
    parser.add_argument("--num_render_images", type=int, default=19,
                        help='the number of render images')
    parser.add_argument("--num_extract_poses", type=int, default=19,
                        help='the number of poses extracted')
    parser.add_argument("--ndc", type=bool, default=True,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--render_height", type=int, default=0,
                        help='the height of image for rendering')
    parser.add_argument("--render_width", type=int, default=0,
                        help='the width of image for rendering')
    parser.add_argument("--render_fx", type=float, default=0,
                        help='focal length in x-axis for rendering')
    parser.add_argument("--render_fy", type=float, default=0,
                        help='focal length in y-axis for rendering')
    parser.add_argument("--render_cx", type=float, default=0,
                        help='optical center in x-axis for rendering')
    parser.add_argument("--render_cy", type=float, default=0,
                        help='optical center in y-axis for rendering')
 
    ## optimization options    
    parser.add_argument("--optimize_nerf", action='store_true',
                        help='whether to optimize NeRF network')
    parser.add_argument("--optimize_pose", action='store_true',
                        help='whether to optimize camera pose')
    parser.add_argument("--optimize_trans", action='store_true',
                        help='whether to optimize transform between RGB camera and Event camera')
    parser.add_argument("--optimize_rgb_crf", action='store_true',
                        help = "whether to optimize response function for RGB camera")
    parser.add_argument("--optimize_event_crf", action='store_true',
                        help = "whether to optimize response function for Event camera")
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate of NeRF')
    parser.add_argument("--pose_lrate", type=float, default=1e-3,
                        help='learning rate of camera pose')
    parser.add_argument("--transform_lrate", type=float, default=1e-6,
                        help='learning rate of the transform between Event camera and RGB camera')
    parser.add_argument("--rgb_crf_lrate", type = float, default = 5e-4,
                        help = "learning rate of rgb_crf")
    parser.add_argument("--event_crf_lrate", type = float, default = 5e-4,
                        help = "learning rate of event_crf")
    parser.add_argument("--decay_rate", type=float, default=0.1,
                        help='learning rate decay of NeRF')
    parser.add_argument("--decay_rate_pose", type=float, default=0.01,
                        help='learning rate decay of RGB camera pose')
    parser.add_argument("--decay_rate_transform", type=float, default=0.01,
                        help='learning rate decay of the transform between event camera and RGB camera')
    parser.add_argument("--decay_rate_rgb_crf", type = float, default = 0.1,
                        help = "learning rate decay of rgb_crf")
    parser.add_argument("--decay_rate_event_crf", type = float, default = 0.1,
                        help = "learning rate decay of event_crf")
    parser.add_argument("--lrate_decay", type=int, default=200,
                        help='exponential learning rate decay (in 1000 steps)')

    ## camera paramaters
    parser.add_argument("--rgb_fx", type=float, default=548.409,
                        help='focal length of RGB camera in x-axis')
    parser.add_argument("--rgb_fy", type=float, default=548.409,
                        help='focal length of RGB camera in y-axis')
    parser.add_argument("--rgb_cx", type=float, default=384.,
                        help='optical center of RGB camera in x-axis')
    parser.add_argument("--rgb_cy", type=float, default=240.,
                        help='optical center of RGB camera in y-axis')
    parser.add_argument("--rgb_width", type=float, default=240.,
                        help='the width of RGB camera image')
    parser.add_argument("--rgb_height", type=float, default=240.,
                        help='the height of RGB camera image')
    parser.add_argument("--rgb_dist", type=float, action="append",
                        help='distortion parameters of RGB camera')
    parser.add_argument("--event_fx", type=float, default=548.409,
                        help='focal length of Event camera in x-axis')
    parser.add_argument("--event_fy", type=float, default=548.409,
                        help='focal length of Event camera in y-axis')
    parser.add_argument("--event_cx", type=float, default=384.,
                        help='optical center of Event camera in x-axis')
    parser.add_argument("--event_cy", type=float, default=240.,
                        help='optical center of Event camera in y-axis')
    parser.add_argument("--event_width", type=int, default=480,
                        help='the width of Event camera image')
    parser.add_argument("--event_height", type=int, default=768,
                        help='the height of Event camera image')
    parser.add_argument("--event_dist", type=float, action="append",
                        help='distortion parameters of Event camera')
    
    ## event stream parameters
    parser.add_argument("--event_threshold", type=float, default=0.1,
                        help='threshold set for events spiking')
    parser.add_argument("--event_shift_start", type=float, default=5,
                        help='shift the start timestamp for event stream')
    parser.add_argument("--event_shift_end", type=float, default=5,
                        help='shift the end timestamp for event stream')
    parser.add_argument("--accumulate_time_length", type=float, default=0.1,
                        help='the percentage of the window')
    parser.add_argument("--random_sampling_window", action='store_true',
                        help='whether to use fixed windows or sliding window')
    parser.add_argument("--event_time_window", action='store_true',
                        help='whether to use fixed windows or sliding window')

    ## logging/saving options
    parser.add_argument("--max_iter", type=int, default=200000,
                        help='maximum number of training iterations')
    parser.add_argument("--console_log_iter", type=int, default=100,
                        help='number of iterations for printing logs on console')
    parser.add_argument("--render_image_iter", type=int, default=25000,
                        help='number of iterations for rendering image for test')
    parser.add_argument("--save_model_iter", type=int, default=10000,
                        help='number of iterations for saving model checkpoint')
    parser.add_argument("--render_video_iter", type=int, default=50000,
                        help='number of iterations for rendering video for test')

    # loss options
    parser.add_argument("--rgb_loss", action='store_true',
                        help='whether to compute RGB image loss')
    parser.add_argument("--event_loss", action='store_true',
                        help='whether to compute event stream loss')
    parser.add_argument("--event_coeff_syn", type=float, default=1.0,
                        help='coefficient for event stream loss on synthetic dataset')
    parser.add_argument("--event_coeff_real", type=float, default=1.0,
                        help='coefficient for event stream loss on real-world dataset')
    parser.add_argument("--rgb_coeff", type=float, default=1.0,
                        help='coefficient for RGB image loss')
    
    # parser.add_argument("--weight_iter", type=int, default=10000,
    #                     help='weight_iter')
    return parser