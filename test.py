import os
import torch
import random
import imageio
import numpy as np

from model import optimize
from utils import img_utils
from utils import pose_utils
from config import config_parser
from undistort import UndistortFisheyeCamera
from run_nerf_helpers import render_image_test, render_video_test

def test(args):
    # calibration parameters dict
    img_calib = {
        "fx": args.rgb_fx, "fy": args.rgb_fy, "cx": args.rgb_cx, "cy": args.rgb_cy,
        "k1": args.rgb_dist[0], "k2": args.rgb_dist[1], "k3": args.rgb_dist[2], "k4": args.rgb_dist[3],
    }
    evt_calib = {
        "fx": args.event_fx, "fy": args.event_fy, "cx": args.event_cx, "cy": args.event_cy,
        "k1": args.event_dist[0], "k2": args.event_dist[1], "k3": args.event_dist[2], "k4": args.event_dist[3],
    }

    print(f"distortion coefficients of rgb camera: \n{args.rgb_dist[0],args.rgb_dist[1],args.rgb_dist[2],args.rgb_dist[3]}\n")
    print(f"distortion coefficients of evt camera: \n{args.event_dist[0],args.event_dist[1],args.event_dist[2],args.event_dist[3]}\n")

    # create undistorter
    img_xy_remap = np.array([])
    evt_xy_remap = np.array([])
    if args.dataset == "TUM_VIE":    
        undistorter = UndistortFisheyeCamera.KannalaBrandt(img_calib, evt_calib)
        # lookup table
        img_xy_remap = undistorter.UndistortImageCoordinate(args.rgb_width, args.rgb_height)
        evt_xy_remap = undistorter.UndistortStreamEventsCoordinate(args.event_width, args.event_height)
    print("shape of image remap", img_xy_remap.shape)
    print("shape of event remap", evt_xy_remap.shape)

    # rgb camera intrinsic matrix
    K_rgb = np.array([
        [img_calib["fx"], 0, img_calib["cx"]], 
        [0, img_calib["fy"], img_calib["cy"]], 
        [0, 0, 1]], dtype = np.float32
    )

    # event camera intrinsic matrix
    K_event = np.array([
        [evt_calib["fx"], 0, evt_calib["cx"]],
        [0, evt_calib["fy"], evt_calib["cy"]],
        [0, 0, 1]], dtype = np.float32
    )

    # camera for rendering
    K_render = np.array([
        [args.render_fx, 0, args.render_cx],
        [0, args.render_fy, args.render_cy],
        [0, 0, 1]], dtype = np.float32
    )

    H_render = args.render_height
    W_render = args.render_width
    if args.render_height == 0 and args.render_width == 0:
        K_render = K_rgb
        H_render = int(args.rgb_height)
        W_render = int(args.rgb_width)

    print("hight of render image", H_render)
    print("weight of render image", W_render)
    print(f"rgb camera intrinsic parameters: \n{K_rgb}\n")
    print(f"event camera intrinsic parameters: \n{K_event}\n")
    print(f"render camera intrinsic parameters: \n{K_render}\n")

    # Create log dir and copy the config file
    logdir = os.path.join(os.path.expanduser(args.logdir), str(args.index))
    testdir = os.path.join(logdir, "test_results")
    os.makedirs(testdir, exist_ok=True)
    # f = os.path.join(logdir, "args.txt")
    # with open(f, "w") as file:
    #     for arg in sorted(vars(args)):
    #         attr = getattr(args, arg)
    #         file.write("{} = {}\n".format(arg, attr))
    # if args.config is not None:
    #     f = os.path.join(logdir, "config.txt")
    #     with open(f, "w") as file:
    #         file.write(open(args.config, "r").read())

    # choose model
    if args.model == "benerf":
        model = optimize.Model(args)
    else:
        print("[Warning] Unknown model type")
        return
    print(f"[INFO] Use model type: {args.model}")

    # Load checkpoint of model
    graph = model.build_network(args)
    optimizer_nerf, optimizer_pose, optimizer_trans, optimizer_rgb_crf, optimizer_event_crf = model.setup_optimizer(args)
    path = os.path.join(logdir, "{:06d}.tar".format(args.checkpoint))
    graph_ckpt = torch.load(path)
    
    graph.load_state_dict(graph_ckpt["graph"])
    optimizer_nerf.load_state_dict(graph_ckpt["optimizer_nerf"])
    optimizer_pose.load_state_dict(graph_ckpt["optimizer_pose"])
    optimizer_trans.load_state_dict(graph_ckpt["optimizer_trans"])
    optimizer_rgb_crf.load_state_dict(graph_ckpt["optimizer_rgb_crf"])
    optimizer_event_crf.load_state_dict(graph_ckpt["optimizer_event_crf"])
    global_step = graph_ckpt["global_step"]
    print("[INFO] Model Load Done!")

    # save poses for test
    if args.extract_poses and global_step > 0:
        extract_poses = graph.get_pose_rgb(args, [0,1], seg_num = args.num_extract_poses)
        pose_utils.save_poses_as_kitti_format(global_step, testdir, extract_poses)
        print("[INFO] Successfully extract camera poses.")
    # render images for test
    if args.render_images and global_step > 0:
        render_images_poses = graph.get_pose_rgb(args, [0,1], seg_num = args.num_render_images)  
        with torch.no_grad(): 
            imgs, depth = render_image_test(
                global_step, graph, render_images_poses, H_render, W_render, K_render, args, testdir, img_xy_remap,
                dir = "image_test", need_depth = args.depth,
            )
            assert len(imgs) > 0, f"[ERROR] Can't successfully render images."
            print("[INFO] Successfully render images.")
    # render video for test
    if args.render_video and global_step > 0:
        render_video_poses = graph.get_pose_rgb(args, [0,1], 90)
        with torch.no_grad():
            rgbs, disps = render_video_test(global_step, graph, render_video_poses, H_render, W_render, K_render, args, img_xy_remap)
            assert len(rgbs) > 0 and len(disps) > 0, f"[ERROR] Can't successfully render video."
        moviebase = os.path.join(testdir, "{}_spiral_{:06d}_".format(args.index, global_step))
        imageio.mimsave(moviebase + "rgb.mp4", img_utils.to8bit(rgbs), fps = 30, quality = 8)
        # imageio.mimsave(moviebase + 'radience.mp4', radiences, fps = 30, quality = 8)
        # imageio.mimsave(moviebase + "disp.mp4", img_utils.to8bit(disps / np.max(disps)), fps = 30, quality = 8)
        print("[INFO] Successfully render video.")

if __name__ == '__main__':
    # load config
    print("[INFO] Loading config...")
    parser = config_parser()
    args = parser.parse_args()

    # setup seed (for exp)
    # torch.set_default_dtype(torch.float32)
    # torch.set_default_device('cuda')
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    os.environ["PYTHONHASHSEED"] = str(0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.random.manual_seed(args.seed)
    if not args.debug:
        # performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # setup device
    print(f"[INFO] Use device: {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # test
    print("[INFO] Start testing...")
    test(args=args)
