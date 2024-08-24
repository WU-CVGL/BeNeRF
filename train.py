import os
import random
import imageio
import torch.nn

from model.nerf import *
from loss import imgloss
from model import optimize
from utils import img_utils
from tqdm import trange, tqdm
from load_data import load_data
from config import config_parser
from metrics import compute_img_metric
from utils.math_utils import rgb2brightlog
from logger.wandb_logger import WandbLogger
from undistort import UndistortFisheyeCamera
from utils.pose_utils import save_poses_as_kitti_format
from run_nerf_helpers import init_nerf, render_image_test, render_video_test

def train(args):
    # start wandb
    logger = WandbLogger(args)

    # loss
    mse_loss = imgloss.MSELoss()
    rgb2gray = img_utils.RGB2Gray()

    print("Loading data...")
    # imgtest: groundtruth shape image
    events, img, imgtest, rgb_exp_ts, poses_ts, poses, ev_poses, trans = load_data(
        args.datadir, args, load_pose = args.loadpose, load_trans = args.loadtrans,
        cubic = "cubic" in args.model, datasource = args.dataset,
    )
    print("Load data successfully!!")

    print("exposure time of rgb image", rgb_exp_ts)
    print(f"Loaded data from {args.datadir}")
    print(f"Loaded image idx: {args.index}")
    print(f"Loaded image size: {img.shape}")
    print(f"Loaded RGB camera pose: {poses}")
    print(f"Loaded Event camera pose: {ev_poses}")
    print(f"Loaded camera Transform: {trans}")

    if trans is not None and ev_poses is None:
        ev_poses = trans

    # Cast intrinsics to right types
    # rgb camera
    H, W = img[0].shape[0], img[0].shape[1]
    H, W = int(H), int(W)

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
        img_xy_remap = undistorter.UndistortImageCoordinate(W, H)
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
        H_render = H
        W_render = W

    print("hight of render image", H_render)
    print("weight of render image", W_render)
    print(f"rgb camera intrinsic parameters: \n{K_rgb}\n")
    print(f"event camera intrinsic parameters: \n{K_event}\n")
    print(f"render camera intrinsic parameters: \n{K_render}\n")

    # Create log dir and copy the config file
    logdir = os.path.join(os.path.expanduser(args.logdir), str(args.index))
    os.makedirs(logdir, exist_ok=True)
    f = os.path.join(logdir, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        f = os.path.join(logdir, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read())

    # choose model
    if args.model == "benerf":
        model = optimize.Model(args)
    else:
        print("Unknown model type")
        return
    print(f"Use model type {args.model}")

    # init model
    if args.load_checkpoint:
        graph = model.build_network(args)
        optimizer, optimizer_pose, optimizer_trans, optimizer_rgb_crf, optimizer_event_crf = model.setup_optimizer(args)
        path = os.path.join(logdir, "{:06d}.tar".format(args.weight_iter))
        graph_ckpt = torch.load(path)

        graph.load_state_dict(graph_ckpt["graph"])
        optimizer.load_state_dict(graph_ckpt["optimizer"])
        optimizer_pose.load_state_dict(graph_ckpt["optimizer_pose"])
        optimizer_trans.load_state_dict(graph_ckpt["optimizer_trans"])
        if args.two_phase:
            global_step = 1
        else:
            global_step = graph_ckpt["global_step"]

        print("Model Load Done!")
    else:
        graph = model.build_network(args, poses = poses, event_poses = ev_poses)

        (
            optimizer_nerf,
            optimizer_pose,
            optimizer_trans,
            optimizer_rgb_crf,
            optimizer_event_crf,
        ) = model.setup_optimizer(args)

        print("No pre-trained weights are used!")
        # initial optimizer
        optimizer_nerf.zero_grad()
        optimizer_pose.zero_grad()
        optimizer_trans.zero_grad()
        optimizer_rgb_crf.zero_grad()
        optimizer_event_crf.zero_grad()

    print("Training is executed...")
    N_iters = args.max_iter + 1
    # N_iters = args.max_iter

    start = 0
    if not args.load_checkpoint:
        global_step = start
    global_step_ = global_step

    for i in trange(start, N_iters):
        i = i + global_step_          
        if i == 0:
            # init weights of nn using Xavier value
            init_nerf(graph.nerf)
            init_nerf(graph.nerf_fine)

        # interpolate poses, ETA and render
        ret_event, ret_rgb, ray_idx_event, ray_idx_rgb, events_accu = graph.forward(
            i, events, rgb_exp_ts, H, W, K_rgb, K_event, args, img_xy_remap, evt_xy_remap 
        )
        pixels_num = ray_idx_event.shape[0]

        # render results of event(start and end)
        ret_gray1 = {
            "rgb_map": ret_event["rgb_map"][:pixels_num],
            "rgb0": ret_event["rgb0"][:pixels_num],
        }
        ret_gray2 = {
            "rgb_map": ret_event["rgb_map"][pixels_num:],
            "rgb0": ret_event["rgb0"][pixels_num:],
        }
        # render results of rgb(contain N sharp image)
        ret_rgb = {"rgb_map": ret_rgb["rgb_map"], "rgb0": ret_rgb["rgb0"]}
        # observed eta
        target_s = events_accu.reshape(-1, 1)[ray_idx_event]

        # use crf 
        if args.optimize_event_crf:
            ret_gray1_fine = graph.event_crf.forward(ret_gray1["rgb_map"])
            ret_gray1_coarse = graph.event_crf.forward(ret_gray1["rgb0"])

            ret_gray2_fine = graph.event_crf.forward(ret_gray2["rgb_map"])
            ret_gray2_coarse = graph.event_crf.forward(ret_gray2["rgb0"])

            ret_gray1 = {"rgb_map": ret_gray1_fine, "rgb0": ret_gray1_coarse}
            ret_gray2 = {"rgb_map": ret_gray2_fine, "rgb0": ret_gray2_coarse}
        if args.optimize_rgb_crf:
            ret_rgb_fine = graph.rgb_crf.forward(ret_rgb["rgb_map"])
            ret_rgb_coarse = graph.rgb_crf.forward(ret_rgb["rgb0"])
            ret_rgb = {"rgb_map": ret_rgb_fine, "rgb0": ret_rgb_coarse}

        # zero grad
        optimizer_nerf.zero_grad()
        optimizer_pose.zero_grad()
        optimizer_trans.zero_grad()
        optimizer_rgb_crf.zero_grad()
        optimizer_event_crf.zero_grad()

        # compute loss
        loss = 0

        # Event loss
        if args.event_loss:
            # Synthetic dataset
            if args.event_threshold > 0:
                # compute acc * C
                target_s *= torch.tensor(args.event_threshold)
                if args.channels == 3:
                    fine_bright2 = rgb2brightlog(rgb2gray(ret_gray2["rgb_map"]), args.dataset)
                    fine_bright1 = rgb2brightlog(rgb2gray(ret_gray1["rgb_map"]), args.dataset)
                    event_loss_fine = mse_loss((fine_bright2 - fine_bright1), target_s)
                else:
                    fine_bright2 = rgb2brightlog(ret_gray2["rgb_map"], args.dataset)
                    fine_bright1 = rgb2brightlog(ret_gray1["rgb_map"], args.dataset)
                    event_loss_fine = mse_loss((fine_bright2 - fine_bright1), target_s)
                event_loss_fine *= args.event_coeff_syn
                logger.write("train_event_loss_fine", event_loss_fine.item())

                if "rgb0" in ret_event:
                    if args.channels == 3:
                        coarse_bright2 = rgb2brightlog(rgb2gray(ret_gray2["rgb0"]), args.dataset)
                        coarse_bright1 = rgb2brightlog(rgb2gray(ret_gray1["rgb0"]), args.dataset)
                        event_loss_coarse = mse_loss((coarse_bright2 - coarse_bright1), target_s)
                    else:
                        coarse_bright2 = rgb2brightlog(ret_gray2["rgb0"], args.dataset)
                        coarse_bright1 = rgb2brightlog(ret_gray1["rgb0"], args.dataset)
                        event_loss_coarse = mse_loss((coarse_bright2 - coarse_bright1), target_s)
                    event_loss_coarse *= args.event_coeff_syn
                    logger.write("train_event_loss_coarse", event_loss_coarse.item())

                # coarse + fine
                event_loss = event_loss_coarse + event_loss_fine
                logger.write("train_event_loss", event_loss.item())
                loss += event_loss
            # Real dataset
            else:
                if args.channels == 3:
                    fine_bright2 = rgb2brightlog(rgb2gray(ret_gray2["rgb_map"]), args.dataset)
                    fine_bright1 = rgb2brightlog(rgb2gray(ret_gray1["rgb_map"]), args.dataset)
                    render_brightness_diff = fine_bright2 - fine_bright1
                    render_norm = render_brightness_diff / (
                        torch.linalg.norm(render_brightness_diff, dim=0, keepdim=True) + 1e-9
                    )
                    target_s_norm = target_s / (
                        torch.linalg.norm(target_s, dim=0, keepdim=True) + 1e-9
                    )
                    event_loss_fine = mse_loss(render_norm, target_s_norm)
                else:
                    fine_bright2 = rgb2brightlog(ret_gray2["rgb_map"], args.dataset)
                    fine_bright1 = rgb2brightlog(ret_gray1["rgb_map"], args.dataset)
                    render_brightness_diff = fine_bright2 - fine_bright1
                    render_norm = render_brightness_diff / (
                        torch.linalg.norm(render_brightness_diff, dim=0, keepdim=True) + 1e-9
                    )
                    target_s_norm = target_s / (
                        torch.linalg.norm(target_s, dim=0, keepdim=True) + 1e-9
                    )
                    event_loss_fine = mse_loss(render_norm, target_s_norm)
                event_loss_fine *= args.event_coeff_real
                logger.write("train_event_loss_fine", event_loss_fine.item())

                if "rgb0" in ret_event:
                    if args.channels == 3:
                        coarse_bright2 = rgb2brightlog(rgb2gray(ret_gray2["rgb0"]), args.dataset)
                        coarse_bright1 = rgb2brightlog(rgb2gray(ret_gray1["rgb0"]), args.dataset)
                        render_brightness_diff = coarse_bright2 - coarse_bright1
                        render_norm = render_brightness_diff / (
                            torch.linalg.norm(render_brightness_diff, dim=0, keepdim=True) + 1e-9
                        )
                        target_s_norm = target_s / (
                            torch.linalg.norm(target_s, dim=0, keepdim=True) + 1e-9
                        )
                        event_loss_coarse = mse_loss(render_norm, target_s_norm)
                    else:
                        coarse_bright2 = rgb2brightlog(ret_gray2["rgb0"], args.dataset)
                        coarse_bright1 = rgb2brightlog(ret_gray1["rgb0"], args.dataset)
                        render_brightness_diff = coarse_bright2 - coarse_bright1
                        render_norm = render_brightness_diff / (
                            torch.linalg.norm(render_brightness_diff, dim=0, keepdim=True) + 1e-9
                        )
                        target_s_norm = target_s / (
                            torch.linalg.norm(target_s, dim=0, keepdim=True) + 1e-9
                        )
                        event_loss_coarse = mse_loss(render_norm, target_s_norm)
                    event_loss_coarse *= args.event_coeff_real
                    logger.write("train_event_loss_coarse", event_loss_coarse.item())
                event_loss = event_loss_coarse + event_loss_fine
                logger.write("train_event_loss", event_loss.item())

                loss += event_loss
        else:
            event_loss = torch.tensor(0)
            event_loss_fine = torch.tensor(0)
            event_loss_coarse = torch.tensor(0)

        # RGB loss
        if args.rgb_loss:
            image = torch.Tensor(img[0])
            target_s = image.reshape(-1, H * W, args.channels)
            target_s = target_s[:, ray_idx_rgb].reshape(-1, args.channels)
            interval = target_s.shape[0]
            synthesized_blur_rgb = 0
            synthesized_blur_rgb0 = 0

            # accumulate sharp RGB images to one blur RGB image
            for j in range(0, args.num_interpolated_pose):
                ray_rgb = ret_rgb["rgb_map"][j * interval : (j + 1) * interval]
                synthesized_blur_rgb += ray_rgb
                if "rgb0" in ret_rgb:
                    ray_extras = ret_rgb["rgb0"][j * interval : (j + 1) * interval]
                    synthesized_blur_rgb0 += ray_extras

                if (j + 1) % args.num_interpolated_pose == 0:
                    synthesized_blur_rgb = synthesized_blur_rgb / args.num_interpolated_pose
                    if "rgb0" in ret_rgb:
                        synthesized_blur_rgb0 = synthesized_blur_rgb0 / args.num_interpolated_pose

            # rgb loss
            rgb_loss_fine = mse_loss(synthesized_blur_rgb, target_s)
            rgb_loss_fine *= args.rgb_coeff
            logger.write("train_rgb_loss_fine", rgb_loss_fine.item())
            if "rgb0" in ret_rgb:
                rgb_loss_coarse = mse_loss(synthesized_blur_rgb0, target_s)
                rgb_loss_coarse *= args.rgb_coeff
                logger.write("train_rgb_loss_coarse", rgb_loss_coarse.item())
            rgb_loss = rgb_loss_fine + rgb_loss_coarse
            logger.write("train_rgb_loss", rgb_loss)

            loss += rgb_loss
        else:
            rgb_loss = torch.tensor(0)
            rgb_loss_fine = torch.tensor(0)
            rgb_loss_coarse = torch.tensor(0)

        logger.write("train_loss", loss.item())

        # backwawrd
        loss.backward()

        # step
        if args.optimize_nerf:
            optimizer_nerf.step()
        if args.optimize_pose:
            optimizer_pose.step()
        if args.optimize_trans:
            optimizer_trans.step()
        if args.optimize_rgb_crf:
            optimizer_rgb_crf.step()
        if args.optimize_event_crf:
            optimizer_event_crf.step()

        # update learning rate
        decay_rate = args.decay_rate
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (
            decay_rate ** (global_step / decay_steps)
        )
        #logger.write("lr_nerf", new_lrate)
        for param_group in optimizer_nerf.param_groups:
            param_group["lr"] = new_lrate

        decay_rate_pose = args.decay_rate_pose
        new_lrate_pose = args.pose_lrate * (
            decay_rate_pose ** (global_step / decay_steps)
        )
        #logger.write("lr_pose", new_lrate_pose)
        for param_group in optimizer_pose.param_groups:
            param_group["lr"] = new_lrate_pose

        decay_rate_transform = args.decay_rate_transform
        new_lrate_trans = args.transform_lrate * (
            decay_rate_transform ** (global_step / decay_steps)
        )
        #logger.write("lr_trans", new_lrate_trans)
        for param_group in optimizer_trans.param_groups:
            param_group["lr"] = new_lrate_trans

        decay_rate_rgb_crf = args.decay_rate_rgb_crf
        new_lrate_rgb_crf = args.rgb_crf_lrate * (
            decay_rate_rgb_crf ** (global_step / decay_steps)
        )
        #logger.write("lr_rgb_crf", new_lrate_rgb_crf)
        for param_group in optimizer_rgb_crf.param_groups:
            param_group["lr"] = new_lrate_rgb_crf

        deacy_rate_event_crf = args.decay_rate_event_crf
        new_lrate_event_crf = args.event_crf_lrate * (
            deacy_rate_event_crf ** (global_step / decay_steps)
        )
        #logger.write("lr_event_crf", new_lrate_event_crf)
        for param_group in optimizer_event_crf.param_groups:
            param_group["lr"] = new_lrate_event_crf

        # print result in console
        if i % args.console_log_iter == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {loss.item()}, event_loss: {event_loss.item()}, rgb_loss: {rgb_loss.item()}, "
                f"event_loss_fine: {event_loss_fine.item()}, event_loss_coarse: {event_loss_coarse.item()}, "
                f"rgb_loss_fine: {rgb_loss_fine.item()}, rgb_loss_coarse: {rgb_loss_coarse.item()}"
            )
        # render image for testing
        if i % args.render_image_iter == 0 and i > 0:
            save_poses = graph.get_pose_rgb(args, rgb_exp_ts, seg_num = args.num_interpolated_pose)
            save_poses_as_kitti_format(i, logdir, save_poses)
            test_poses = graph.get_pose_rgb(args, rgb_exp_ts, seg_num = args.num_interpolated_pose)

            with torch.no_grad():
                if args.load_checkpoint == False:
                    img_test_dir  = "images_test"
                else:
                    img_test_dir = "results"
                    
                imgs, depth = render_image_test(
                    i, graph, test_poses, H_render, W_render, K_render, args, logdir, img_xy_remap,
                    dir = img_test_dir, need_depth = args.depth,
                )

                if len(imgs) > 0:
                    logger.write_img("test_img_mid", imgs[len(imgs) // 2])
                    logger.write_imgs("test_img_all", imgs)
                    # logger.write_img("test_radience_mid", radiences[len(radiences) // 2])
                    # logger.write_imgs("test_radience_all", radiences)
                    if args.dataset in ["BeNeRF_Unreal", "BeNeRF_Blender", "E2NeRF_Synthetic"]:
                        imgtest = torch.Tensor(imgtest)
                        img_mid = imgs[len(imgs) // 2] / 255.0
                        img_mid = torch.unsqueeze(torch.tensor(img_mid, dtype=torch.float32), dim=0)

                        test_mid_psnr = compute_img_metric(img_mid, imgtest, metric="psnr")
                        test_mid_ssim = compute_img_metric(img_mid, imgtest, metric="ssim")
                        test_mid_lpips = compute_img_metric(img_mid, imgtest, metric="lpips")

                        logger.write("test_mid_psnr", test_mid_psnr)
                        logger.write("test_mid_ssim", test_mid_ssim)
                        logger.write("test_mid_lpips", test_mid_lpips)
                if len(depth) > 0:
                    pass
        # render video for test
        if i % args.render_video_iter == 0 and i > 0 and not args.load_checkpoint:
            render_poses = graph.get_pose_rgb(args, rgb_exp_ts, 90)
            with torch.no_grad():
                rgbs, disps = render_video_test(i, graph, render_poses, H_render, W_render, K_render, args, img_xy_remap)
            print("Done, saving", rgbs.shape, disps.shape)
            moviebase = os.path.join(logdir, "{}_spiral_{:06d}_".format(args.index, i))
            imageio.mimsave(moviebase + "rgb.mp4", img_utils.to8bit(rgbs), fps = 30, quality = 8)
            # imageio.mimsave(moviebase + 'radience.mp4', radiences, fps = 30, quality = 8)
            imageio.mimsave(moviebase + "disp.mp4", img_utils.to8bit(disps / np.max(disps)), fps = 30, quality = 8)
        # save checkpoint
        if i % args.save_model_iter == 0 and i > 0:
            path = os.path.join(logdir, "{:06d}.tar".format(i))
            torch.save({
                "global_step": global_step,
                "graph": graph.state_dict(),
                "optimizer_nerf": optimizer_nerf.state_dict(),
                "optimizer_pose": optimizer_pose.state_dict(),
                "optimizer_trans": optimizer_trans.state_dict(),
                "optimizer_rgb_crf": optimizer_rgb_crf.state_dict(),
                "optimizer_event_crf": optimizer_event_crf.state_dict()}, 
                path
            )
            # save to logger
            #logger.write_checkpoint(path, args.expname)
            print("Saved checkpoints at", path)

        logger.update_buffer()

        if args.load_checkpoint:
            break
        global_step += 1

    # after train callback
    model.after_train()

if __name__ == "__main__":
    # load config
    print("Loading config")
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
    print(f"Use device: {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # train
    print("Start training...")
    train(args=args)
