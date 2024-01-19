import os
import random

import imageio
import torch.nn
from tqdm import trange, tqdm

from config import config_parser
from load_data import load_data
from logger.wandb_logger import WandbLogger
from loss import imgloss
from metrics import compute_img_metric
from model import nerf_cubic_optimpose
from model import nerf_cubic_optimexposure
from model import nerf_cubic_optimtrans_event
from model import nerf_cubic_optimposeset
from model import nerf_cubic_optimtrans
from model import nerf_cubic_rigidtrans
from model import nerf_linear_optimpose
from model import nerf_linear_optimposeset
from model import nerf_linear_optimtrans
from model import test_model
from model.nerf import *
from run_nerf_helpers import init_nerf, render_image_test, render_video_test
from utils import img_utils
from utils.math_utils import safelog
from undistort import UndistortFisheyeCamera



def train(args):
    # Load data images are groundtruth
    logger = WandbLogger(args)

    # transforms
    mse_loss = imgloss.MSELoss()
    rgb2gray = img_utils.RGB2Gray()

    print("Loading data...")

    # imgtests:for render test
    events, images, imgtests, poses_ts, poses, ev_poses, trans = load_data(
        args.datadir,
        args,
        load_pose = args.loadpose,
        load_trans = args.loadtrans,
        cubic = "cubic" in args.model,
        datasource = args.dataset,
    )

    print(f"Loaded data from {args.datadir}")
    print(f"Loaded image idx: {args.idx}")
    print(f"Loaded image size: {images.shape}")
    print(f"Camera Pose: {poses}")
    print(f"Event Camera Pose: {ev_poses}")
    print(f"Camera Trans: {trans}")

    if trans is not None and ev_poses is None:
        ev_poses = trans

    # calibration parameters dict
    img_calib = {
        "fx": args.focal_x,
        "fy": args.focal_y,
        "cx": args.cx,
        "cy": args.cy,
        "k1": args.img_dist[0],
        "k2": args.img_dist[1],
        "k3": args.img_dist[2],
        "k4": args.img_dist[3],
    }
    evt_calib = {
        "fx": args.focal_event_x,
        "fy": args.focal_event_y,
        "cx": args.event_cx,
        "cy": args.event_cy,
        "k1": args.evt_dist[0],
        "k2": args.evt_dist[1],
        "k3": args.evt_dist[2],
        "k4": args.evt_dist[3],
    }

    # Cast intrinsics to right types
    # rgb camera
    H, W = images[0].shape[0], images[0].shape[1]
    H, W = int(H), int(W)

    # intrinsic matrix
    K = torch.Tensor(
        [
            [img_calib["fx"], 0, img_calib["cx"]], 
            [0, img_calib["fy"], img_calib["fy"]], 
            [0, 0, 1]
        ]
    )

    # event camera
    K_event = torch.Tensor(
        [
            [evt_calib["fx"], 0, evt_calib["cx"]],
            [0, evt_calib["fy"], evt_calib["fy"]],
            [0, 0, 1],
        ]
    )

    # camera for rendering
    K_render = torch.Tensor(
        [
            [args.render_focal_x, 0, args.render_cx],
            [0, args.render_focal_y, args.render_cy],
            [0, 0, 1],
        ]
    )

    # create undistorter    
    undistorter = UndistortFisheyeCamera.KannalaBrandt(img_calib, evt_calib)

    # undistortion if use TUMVIE dataset
    if args.dataset == "TUMVIE":

        print(f"camera intrinsic parameters before undistortion: \n{K}\n")
        print(f"event camera intrinsic parameters before undistortion:: \n{K_event}\n")
        
        # Get new intrinsic parameters after undistortion
        raw_img_res = np.array([H, W])
        new_img_res = np.array([H, W])
        raw_evt_res = np.array([args.h_event, args.w_event])
        new_evt_res = np.array([args.h_event, args.w_event])
        img_K_new, evt_K_new = undistorter.GetNewIntrinsicMatrix(
            raw_img_res, raw_evt_res, new_img_res, new_evt_res
        )
        K = torch.Tensor(img_K_new)
        K_event = torch.Tensor(evt_K_new)
        
        # Undistort image
        img_dist = images[0].reshape(1024, 1024)
        img_undist = undistorter.UndistortImage(img_dist, img_K_new, new_img_res)
        images[0] = img_undist.reshape((1024, 1024, 1))

    H_render = args.render_h
    W_render = args.render_w
    if args.render_h == 0 and args.render_w == 0:
        K_render = K
        H_render = H
        W_render = W

    print(f"camera intrinsic parameters: \n{K}\n")
    print(f"event camera intrinsic parameters: \n{K_event}\n")
    print(f"render camera intrinsic parameters: \n{K_render}\n")

    # Create log dir and copy the config file
    logdir = os.path.join(os.path.expanduser(args.logdir), args.expname)
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
    if args.model == "cubic_optimpose":
        model = nerf_cubic_optimpose.Model(args)
    elif args.model == "cubic_optimtrans":
        model = nerf_cubic_optimtrans.Model(args)
    elif args.model == "cubic_optimtrans_event":
        model = nerf_cubic_optimtrans_event.Model(args)
    elif args.model == "cubic_optimposeset":
        model = nerf_cubic_optimposeset.Model(args)
    elif args.model == "cubic_optim_exposure":
        model = nerf_cubic_optimexposure.Model(args)
    elif args.model == "cubic_rigidtrans":
        model = nerf_cubic_rigidtrans.Model(args)
    elif args.model == "linear_optimpose":
        model = nerf_linear_optimpose.Model(args)
    elif args.model == "linear_optimtrans":
        model = nerf_linear_optimtrans.Model(args)
    elif args.model == "linear_optimposeset":
        model = nerf_linear_optimposeset.Model(args)
    elif args.model == "test":
        model = test_model.Model(args)
    else:
        print("Unknown model type")
        return

    print(f"Use model type {args.model}")

    # init model
    if args.load_weights:
        graph = model.build_network(args)
        optimizer, optimizer_pose, optimizer_trans = model.setup_optimizer(args)
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
        graph = model.build_network(args, poses=poses, event_poses=ev_poses)

        (
            optimizer,
            optimizer_pose,
            optimizer_trans,
            optimizer_rgb_crf,
            optimizer_event_crf,
        ) = model.setup_optimizer(args)

        print("No pre-trained weights are used!")

    print("Training is executed...")
    N_iters = args.max_iter + 1

    start = 0
    if not args.load_weights:
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
            i, events, H, W, K, K_event, args, undistorter
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

        # use crf for event data
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
        optimizer_pose.zero_grad()
        optimizer_trans.zero_grad()
        optimizer.zero_grad()
        optimizer_rgb_crf.zero_grad()
        optimizer_event_crf.zero_grad()

        # compute loss
        loss = 0

        # Event loss
        # Synthetic dataset
        if args.threshold > 0:
            # compute acc * C
            target_s *= torch.tensor(args.threshold)

            if args.channels == 3:
                img_loss = mse_loss(
                    safelog(rgb2gray(ret_gray2["rgb_map"])) - safelog(rgb2gray(ret_gray1["rgb_map"])),
                    target_s,
                )
            else:
                img_loss = mse_loss(
                    safelog(ret_gray2["rgb_map"]) - safelog(ret_gray1["rgb_map"]),
                    target_s,
                )
            img_loss *= args.event_coefficient
            logger.write("train_event_loss_fine", img_loss.item())

            if "rgb0" in ret_event:
                if args.channels == 3:
                    img_loss0 = mse_loss(
                        safelog(rgb2gray(ret_gray2["rgb0"])) - safelog(rgb2gray(ret_gray1["rgb0"])),
                        target_s,
                    )
                else:
                    img_loss0 = mse_loss(
                        safelog(ret_gray2["rgb0"]) - safelog(ret_gray1["rgb0"]),
                        target_s,
                    )
                img_loss0 *= args.event_coefficient
                logger.write("train_event_loss_coarse", img_loss0.item())

            # coarse + fine
            event_loss = img_loss0 + img_loss

            logger.write("train_event_loss", event_loss.item())

            loss += event_loss
        # Real dataset
        else:
            if args.channels == 3:
                render_brightness_diff = safelog(rgb2gray(ret_gray2["rgb_map"])) - safelog(rgb2gray(ret_gray1["rgb_map"]))
                render_norm = render_brightness_diff / (
                    torch.linalg.norm(render_brightness_diff, dim=0, keepdim=True) + 1e-9
                )
                target_s_norm = target_s / (
                    torch.linalg.norm(target_s, dim=0, keepdim=True) + 1e-9
                )
                img_loss = mse_loss(render_norm, target_s_norm)
            else:
                render_brightness_diff = safelog(ret_gray2["rgb_map"]) - safelog(ret_gray1["rgb_map"])
                render_norm = render_brightness_diff / (
                    torch.linalg.norm(render_brightness_diff, dim=0, keepdim=True) + 1e-9
                )
                target_s_norm = target_s / (
                    torch.linalg.norm(target_s, dim=0, keepdim=True) + 1e-9
                )
                img_loss = mse_loss(render_norm, target_s_norm)
            img_loss *= args.event_coefficient
            img_loss *= args.real_coeff
            logger.write("train_event_loss_fine", img_loss.item())

            if "rgb0" in ret_event:
                if args.channels == 3:
                    render_brightness_diff = safelog(rgb2gray(ret_gray2["rgb0"])) - safelog(rgb2gray(ret_gray1["rgb0"]))
                    render_norm = render_brightness_diff / (
                        torch.linalg.norm(render_brightness_diff, dim=0, keepdim=True) + 1e-9
                    )
                    target_s_norm = target_s / (
                        torch.linalg.norm(target_s, dim=0, keepdim=True) + 1e-9
                    )
                    img_loss0 = mse_loss(render_norm, target_s_norm)
                else:
                    render_brightness_diff = safelog(ret_gray2["rgb0"]) - safelog(ret_gray1["rgb0"])
                    render_norm = render_brightness_diff / (
                        torch.linalg.norm(render_brightness_diff, dim=0, keepdim=True) + 1e-9
                    )
                    target_s_norm = target_s / (
                        torch.linalg.norm(target_s, dim=0, keepdim=True) + 1e-9
                    )
                    img_loss0 = mse_loss(render_norm, target_s_norm)

                img_loss0 *= args.event_coefficient
                img_loss0 *= args.real_coeff
                logger.write("train_event_loss_coarse", img_loss0.item())

            event_loss = img_loss0 + img_loss

            logger.write("train_event_loss", event_loss.item())

            loss += event_loss

        # RGB loss
        if args.rgb_loss:
            image = torch.Tensor(images[0])
            target_s = image.reshape(-1, H * W, args.channels)
            target_s = target_s[:, ray_idx_rgb]
            target_s = target_s.reshape(-1, args.channels)
            interval = target_s.shape[0]
            rgb_ = 0
            extras_ = 0
            blur_loss, extras_blur_loss = 0, 0
            rgb_list = []
            extras_list = []
            for j in range(0, args.deblur_images):
                # accumulate sharp rgb to blur rgb
                ray_rgb = ret_rgb["rgb_map"][j * interval : (j + 1) * interval]
                rgb_ += ray_rgb

                # loss for blur image
                if args.rgb_blur_loss and j % (args.deblur_images // 2) == 0:
                    blur_loss += mse_loss(ray_rgb, target_s)

                if "rgb0" in ret_rgb:
                    ray_extras = ret_rgb["rgb0"][j * interval : (j + 1) * interval]
                    extras_ += ray_extras

                    # loss for blur image
                    if args.rgb_blur_loss and j % (args.deblur_images // 2) == 0:
                        extras_blur_loss += mse_loss(ray_extras, target_s)

                if (j + 1) % args.deblur_images == 0:
                    rgb_ = rgb_ / args.deblur_images
                    rgb_list.append(rgb_)
                    rgb_ = 0
                    if "rgb0" in ret_rgb:
                        extras_ = extras_ / args.deblur_images
                        extras_list.append(extras_)
                        extras_ = 0

            rgb_blur = torch.stack(rgb_list, 0)
            rgb_blur = rgb_blur.reshape(-1, args.channels)

            if "rgb0" in ret_rgb:
                extras_blur = torch.stack(extras_list, 0)
                extras_blur = extras_blur.reshape(-1, args.channels)

            # rgb loss
            rgb_loss_fine = mse_loss(rgb_blur, target_s)
            rgb_loss_fine *= args.rgb_coefficient
            logger.write("train_rgb_loss_fine", rgb_loss_fine.item())

            if "rgb0" in ret_rgb:
                rgb_loss_coarse = mse_loss(extras_blur, target_s)
                rgb_loss_coarse *= args.rgb_coefficient
                logger.write("train_rgb_loss_coarse", rgb_loss_coarse.item())

            rgb_loss = rgb_loss_fine + rgb_loss_coarse
            logger.write("train_rgb_loss", rgb_loss)
            loss += rgb_loss

            # loss for blur image
            if args.rgb_blur_loss:
                blur_loss *= args.rgb_blur_coefficient
                extras_blur_loss *= args.rgb_blur_coefficient
                logger.write("train_rgb_blur_loss_fine", blur_loss.item())
                logger.write("train_rgb_blur_loss_coarse", extras_blur_loss.item())
                rgb_blur_loss = blur_loss + extras_blur_loss
                logger.write("train_rgb_blur_loss", rgb_blur_loss.item())
                loss += rgb_blur_loss
        else:
            rgb_loss = torch.tensor(0)
            rgb_loss_fine = torch.tensor(0)
            rgb_loss_coarse = torch.tensor(0)

        logger.write("train_loss", loss.item())

        # backwawrd
        loss.backward()

        # step
        if args.optimize_nerf:
            optimizer.step()
        if args.optimize_se3:
            optimizer_pose.step()
        if args.optimize_event:
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
        logger.write("lr_nerf", new_lrate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate

        decay_rate_pose = args.decay_rate_pose
        new_lrate_pose = args.pose_lrate * (
            decay_rate_pose ** (global_step / decay_steps)
        )
        logger.write("lr_pose", new_lrate_pose)
        for param_group in optimizer_pose.param_groups:
            param_group["lr"] = new_lrate_pose

        decay_rate_transform = args.decay_rate_transform
        new_lrate_trans = args.transform_lrate * (
            decay_rate_transform ** (global_step / decay_steps)
        )
        logger.write("lr_trans", new_lrate_trans)
        for param_group in optimizer_trans.param_groups:
            param_group["lr"] = new_lrate_trans

        decay_rate_rgb_crf = args.decay_rate_rgb_crf
        new_lrate_rgb_crf = args.rgb_crf_lrate * (
            decay_rate_rgb_crf ** (global_step / decay_steps)
        )
        logger.write("lr_rgb_crf", new_lrate_rgb_crf)
        for param_group in optimizer_rgb_crf.param_groups:
            param_group["lr"] = new_lrate_rgb_crf

        deacy_rate_event_crf = args.decay_rate_event_crf
        new_lrate_event_crf = args.event_crf_lrate * (
            deacy_rate_event_crf ** (global_step / decay_steps)
        )
        logger.write("lr_event_crf", new_lrate_event_crf)
        for param_group in optimizer_event_crf.param_groups:
            param_group["lr"] = new_lrate_event_crf

        # print result in console
        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {loss.item()}, event_loss: {event_loss.item()}, rgb_loss: {rgb_loss.item()}, "
                f"event_fine_loss: {img_loss.item()}, event_coarse_loss: {img_loss0.item()}, "
                f"rgb_loss_fine: {rgb_loss_fine.item()}, rgb_loss_coarse: {rgb_loss_coarse.item()}"
            )

        # save checkpoint
        if i % args.i_weights == 0 and i > 0:
            path = os.path.join(logdir, "{:06d}.tar".format(i))
            torch.save(
                {
                    "global_step": global_step,
                    "graph": graph.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "optimizer_pose": optimizer_pose.state_dict(),
                    "optimizer_trans": optimizer_trans.state_dict(),
                },
                path,
            )
            # save to logger
            logger.write_checkpoint(path, args.expname)
            print("Saved checkpoints at", path)

        # test
        if i % args.i_img == 0 and i > 0:
            test_poses = graph.get_pose_rgb(
                args,
                seg_num = args.deblur_images
                if args.deblur_images % 2 == 1
                else args.deblur_images + 1,
            )

            with torch.no_grad():
                imgs, depth = render_image_test(
                    i,
                    graph,
                    test_poses,
                    H_render,
                    W_render,
                    K_render,
                    args,
                    logdir,
                    dir = "images_test",
                    need_depth = args.depth,
                )

                if len(imgs) > 0:
                    logger.write_img("test_img_mid", imgs[len(imgs) // 2])
                    logger.write_imgs("test_img_all", imgs)
                    # logger.write_img("test_radience_mid", radiences[len(radiences) // 2])
                    # logger.write_imgs("test_radience_all", radiences)
                    if args.dataset == "Unreal" or args.dataset == "Blender":
                        img_mid = imgs[len(imgs) // 2] / 255.0
                        img_mid = torch.unsqueeze(
                            torch.tensor(img_mid, dtype=torch.float32), dim=0
                        )
                        test_mid_psnr = compute_img_metric(
                            img_mid, imgtests, metric="psnr"
                        )
                        # test_mid_ssim = compute_img_metric(img_mid, imgtests, metric="ssim")
                        test_mid_lpips = compute_img_metric(
                            img_mid, imgtests, metric="lpips"
                        )

                        logger.write("test_mid_psnr", test_mid_psnr)
                        # logger.write("test_mid_ssim", test_mid_ssim)
                        logger.write("test_mid_lpips", test_mid_lpips)
                if len(depth) > 0:
                    logger.write_img("test_depth_mid", depth[len(depth) // 2])
                    logger.write_imgs("test_depth_all", depth)

        if i % args.i_video == 0 and i > 0:
            render_poses = graph.get_pose_rgb(args, 90)

            with torch.no_grad():
                rgbs, disps = render_video_test(
                    graph, render_poses, H_render, W_render, K_render, args
                )
            print("Done, saving", rgbs.shape, disps.shape)
            moviebase = os.path.join(
                logdir, "{}_spiral_{:06d}_".format(args.expname, i)
            )
            imageio.mimsave(
                moviebase + "rgb.mp4", img_utils.to8bit(rgbs), fps=30, quality=8
            )
            # imageio.mimsave(moviebase + 'radience.mp4', radiences, fps = 30, quality = 8)
            imageio.mimsave(
                moviebase + "disp.mp4",
                img_utils.to8bit(disps / np.max(disps)),
                fps=30,
                quality=8,
            )

        logger.update_buffer()
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
