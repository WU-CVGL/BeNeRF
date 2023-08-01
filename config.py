import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()

    # device
    parser.add_argument("--device", type=int, default=0,
                        help='cuda id to use')

    # others
    parser.add_argument('--config', is_config_file=True, default='./configs/testconfig.txt',
                        help='config file path')
    parser.add_argument("--project", type=str, default="event-bad-nerf",
                        help='the viewer to use (wandb)')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--deblur_images", type=int, default=7,
                        help='the number of sharp images one blur image corresponds to')
    parser.add_argument("--pixels", type=int, default=320,
                        help='like N_rand')
    parser.add_argument("--skip", type=int, default=8,
                        help='original llffhold before concatenate images')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step) = deblur_images x pixels')

    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate of NeRF')
    parser.add_argument("--pose_lrate", type=float, default=1e-3,
                        help='learning rate of rgb camera pose')
    parser.add_argument("--transform_lrate", type=float, default=1e-6,
                        help='learning rate of the transform between event camera and rgb camera')

    parser.add_argument("--decay_rate", type=float, default=0.1,
                        help='learning rate decay of NeRF')
    parser.add_argument("--decay_rate_pose", type=float, default=0.01,
                        help='learning rate decay of rgb camera pose')
    parser.add_argument("--decay_rate_transform", type=float, default=0.01,
                        help='learning rate decay of the transform between event camera and rgb camera')

    parser.add_argument("--lrate_decay", type=int, default=200,
                        help='exponential learning rate decay (in 1000 steps)')

    parser.add_argument("--chunk", type=int, default=1024 * 2,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 32,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
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

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--ndc", type=bool, default=True,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # idx
    parser.add_argument("--idx", type=int, default=0,
                        help='idx in the dataset to deblur')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--focal", type=float, default=548.409,
                        help='focal length of images')
    parser.add_argument("--dataset_event_split", type=int, default=500,
                        help='tv')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img", type=int, default=25000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=500000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--load_weights", action='store_true',
                        help='frequency of weight ckpt loading')
    parser.add_argument("--weight_iter", type=int, default=10000,
                        help='weight_iter')
    parser.add_argument("--max_iter", type=int, default=200000,
                        help='max_iter')

    # barf: up & down
    parser.add_argument("--barf", action='store_true',
                        help='barf')
    parser.add_argument("--barf_start", type=float, default=0.1,
                        help='barf start')
    parser.add_argument("--barf_end", type=float, default=0.9,
                        help='barf start')

    # tv loss
    parser.add_argument("--tv_loss", action='store_true',
                        help='TV loss')
    parser.add_argument("--tv_width_nerf", type=int, default=15,
                        help='tv')
    parser.add_argument("--tv_loss_rgb", action='store_true',
                        help='tv_rgb')
    parser.add_argument("--tv_loss_gray", action='store_true',
                        help='tv_gray')
    parser.add_argument("--tv_loss_lambda", type=float, default=0.001,
                        help='barf start')
    parser.add_argument("--n_tvloss", type=int, default=0,
                        help='barf start')

    # 分段
    parser.add_argument("--two_phase", action='store_true',
                        help='if use two-phase optimization')

    # optimize
    parser.add_argument("--optimize_se3", action='store_true',
                        help='whether to optimize SE3 network')
    parser.add_argument("--optimize_nerf", action='store_true',
                        help='whether to optimize NeRF network')
    parser.add_argument("--optimize_trans", action='store_true',
                        help='whether to optimize transformation matrix')

    # event parameter
    parser.add_argument("--threshold", type=float, default=0.1,
                        help='threshold set for events spiking')
    parser.add_argument("--rgb_loss", action='store_true',
                        help='')
    parser.add_argument("--rgb_blur_loss", action='store_true',
                        help='')
    parser.add_argument("--channels", type=int, default=3,
                        help='whether to use 3-channel or single-channel images')
    parser.add_argument("--N_pix_no_event", type=int, default=1024,
                        help='number of sampled rays where no events spiking')
    parser.add_argument("--N_pix_event", type=int, default=2048,
                        help='number of sampled rays where with events spiking')
    parser.add_argument("--N_pix_rgb", type=int, default=1024,
                        help='number of sampled rays for computation of image loss function')

    # window
    parser.add_argument("--window_percent", type=float, default=0.1,
                        help='the percentage of the window')
    parser.add_argument("--window_desc", action='store_true',
                        help='turn on windows desc/not')
    parser.add_argument("--window_percent_end", type=float, default=0.05,
                        help='the percentage of the window end (when window_dec enabled)')
    parser.add_argument("--window_desc_end", type=float, default=0.6,
                        help='when iter reach window_desc_end * max_iter, it will maintain window_percent_end')
    parser.add_argument("--random_window", action='store_true',
                        help='whether to use fixed windows or sliding window')
    parser.add_argument("--time_window", action='store_true',
                        help='whether to use fixed windows or sliding window')

    # coefficient for loss
    parser.add_argument("--event_coefficient", type=float, default=1.0,
                        help='coefficient for event loss')
    parser.add_argument("--rgb_coefficient", type=float, default=1.0,
                        help='coefficient for rgb loss')
    parser.add_argument("--rgb_blur_coefficient", type=float, default=0.5,
                        help='coefficient for rgb loss')

    # viewer
    parser.add_argument("--viewer", type=str, default="wandb",
                        help='the viewer to use (wandb)')

    parser.add_argument("--fix_pose", action='store_true',
                        help='the viewer to use (wandb)')
    return parser
