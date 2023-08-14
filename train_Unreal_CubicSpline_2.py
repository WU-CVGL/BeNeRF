import os
import random

import torch.nn
from tqdm import trange, tqdm

import optimize_pose_CubicSpline_2
from config import config_parser
from load_llff import load_llff_data
from logger.wandb_logger import WandbLogger
from loss import imgloss
from nerf import *
from run_nerf_helpers import init_nerf, render_image_test
from utils import imgutils
from utils.mathutils import safelog
from metrics import compute_img_metric


def train(args):
    # Load data images are groundtruth
    logger = WandbLogger(args)

    # transforms
    mse_loss = imgloss.MSELoss()
    rgb2gray = imgutils.RGB2Gray()

    K = None
    poses = None

    if args.dataset_type == 'llff':
        print("Use llff data")
        events, images, imgtests, poses_ts, poses, ev_poses = load_llff_data(args.datadir, factor=args.factor,
                                                                             idx=args.idx,
                                                                             gray=args.channels == 1, load_pose=True,
                                                                             deblur_dataset=args.dataset_event_split)
        print('Loaded llff data', images.shape, args.datadir, args.idx)
    elif args.dataset_type == 'davis':
        print("Use davis data")
        events, images, imgtests, poses_ts = load_davis_data(args.datadir, factor=args.factor, idx=args.idx)
        print('Loaded davis data', images.shape, args.datadir, args.idx)
    elif args.dataset_type == "esim":
        pass
    elif args.dataset_type == "eds":
        pass
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = images[0].shape[0], images[0].shape[1], args.focal
    H, W = int(H), int(W)

    if K is None:
        K = torch.Tensor([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    print('camera intrinsic parameters: ', K, ' !!!')

    # Create log dir and copy the config file
    basedir = os.path.join(os.getcwd(), "logs")
    args.__setattr__("basedir", basedir)
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # init model
    if args.load_weights:
        model = optimize_pose_CubicSpline_2.Model()
        graph = model.build_network(args)
        optimizer, optimizer_pose, optimizer_trans = model.setup_optimizer(args)
        path = os.path.join(basedir, expname, '{:06d}.tar'.format(args.weight_iter))
        graph_ckpt = torch.load(path)

        graph.load_state_dict(graph_ckpt['graph'])
        optimizer.load_state_dict(graph_ckpt['optimizer'])
        optimizer_pose.load_state_dict(graph_ckpt['optimizer_pose'])
        optimizer_trans.load_state_dict(graph_ckpt['optimizer_trans'])
        if args.two_phase:
            global_step = 1
        else:
            global_step = graph_ckpt['global_step']

        print('Model Load Done!')
    else:
        model = optimize_pose_CubicSpline_2.Model()
        graph = model.build_network(args, poses=poses, event_poses=ev_poses)  # nerf, nerf_fine, forward
        optimizer, optimizer_pose, optimizer_trans = model.setup_optimizer(args)
        print('Not Load Model!')

    print('Begin')
    N_iters = args.max_iter + 1

    start = 0
    if not args.load_weights:
        global_step = start
    global_step_ = global_step
    threshold = N_iters + 101

    for i in trange(start, N_iters):
        i = i + global_step_
        if i == 0:
            init_nerf(graph.nerf)
            init_nerf(graph.nerf_fine)

        if i % args.i_video == 0 and i > 0:
            ret_event, ret_rgb, ray_idx_event, ray_idx_rgb, test_poses, events_accu = graph.forward(i, poses_ts,
                                                                                                    threshold, events,
                                                                                                    H, W, K, args)

        elif i % args.i_img == 0 and i > 0:
            ret_event, ret_rgb, ray_idx_event, ray_idx_rgb, test_poses, events_accu = graph.forward(i, poses_ts,
                                                                                                    threshold, events,
                                                                                                    H, W, K, args)
        else:
            ret_event, ret_rgb, ray_idx_event, ray_idx_rgb, events_accu = graph.forward(i, poses_ts, threshold, events,
                                                                                        H, W, K, args)

        pixels_num = ray_idx_event.shape[0]

        ret_gray1 = {'rgb_map': ret_event['rgb_map'][:pixels_num],
                     'rgb0': ret_event['rgb0'][:pixels_num]}
        ret_gray2 = {'rgb_map': ret_event['rgb_map'][pixels_num:],
                     'rgb0': ret_event['rgb0'][pixels_num:]}
        ret_rgb = {'rgb_map': ret_rgb['rgb_map'],
                   'rgb0': ret_rgb['rgb0']}

        target_s = events_accu.reshape(-1, 1)[ray_idx_event]

        # backward
        optimizer_pose.zero_grad()
        optimizer_trans.zero_grad()
        optimizer.zero_grad()
        if args.channels == 3:
            img_loss = mse_loss(safelog(rgb2gray(ret_gray2['rgb_map'])) - safelog(rgb2gray(ret_gray1['rgb_map'])),
                                target_s)
        else:
            img_loss = mse_loss(safelog(ret_gray2['rgb_map']) - safelog(ret_gray1['rgb_map']), target_s)
        img_loss *= args.event_coefficient
        logger.write("train_event_loss_fine", img_loss.item())

        # Event loss
        if 'rgb0' in ret_event:
            if args.channels == 3:
                img_loss0 = mse_loss(safelog(rgb2gray(ret_gray2['rgb0'])) - safelog(rgb2gray(ret_gray1['rgb0'])),
                                     target_s)
            else:
                img_loss0 = mse_loss(safelog(ret_gray2['rgb0']) - safelog(ret_gray1['rgb0']), target_s)
            img_loss0 *= args.event_coefficient
            logger.write("train_event_loss_coarse", img_loss0.item())

        event_loss = img_loss0 + img_loss

        logger.write("train_event_loss", event_loss.item())

        loss = event_loss

        # RGB loss
        if args.rgb_loss:
            target_s = images[0].reshape(-1, H * W, args.channels)
            target_s = target_s[:, ray_idx_rgb]
            target_s = target_s.reshape(-1, args.channels)
            interval = target_s.shape[0]
            rgb_ = 0
            extras_ = 0
            blur_loss, extras_blur_loss = 0, 0
            rgb_list = []
            extras_list = []
            for j in range(0, args.deblur_images):
                ray_rgb = ret_rgb['rgb_map'][j * interval:(j + 1) * interval]
                rgb_ += ray_rgb

                # loss for blur image
                if args.rgb_blur_loss:
                    blur_loss += mse_loss(ray_rgb, target_s)

                if 'rgb0' in ret_rgb:
                    ray_extras = ret_rgb['rgb0'][j * interval:(j + 1) * interval]
                    extras_ += ray_extras

                    # loss for blur image
                    if args.rgb_blur_loss:
                        extras_blur_loss += mse_loss(ray_extras, target_s)

                if (j + 1) % args.deblur_images == 0:
                    rgb_ = rgb_ / args.deblur_images
                    rgb_list.append(rgb_)
                    rgb_ = 0
                    if 'rgb0' in ret_rgb:
                        extras_ = extras_ / args.deblur_images
                        extras_list.append(extras_)
                        extras_ = 0

            rgb_blur = torch.stack(rgb_list, 0)
            rgb_blur = rgb_blur.reshape(-1, args.channels)

            if 'rgb0' in ret_rgb:
                extras_blur = torch.stack(extras_list, 0)
                extras_blur = extras_blur.reshape(-1, args.channels)

            # rgb loss
            rgb_loss_fine = mse_loss(rgb_blur, target_s)
            rgb_loss_fine *= args.rgb_coefficient
            logger.write("train_rgb_loss_fine", rgb_loss_fine.item())

            if 'rgb0' in ret_rgb:
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

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = args.decay_rate
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        logger.write("lr_nerf", new_lrate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        decay_rate_pose = args.decay_rate_pose
        new_lrate_pose = args.pose_lrate * (decay_rate_pose ** (global_step / decay_steps))
        logger.write("lr_pose", new_lrate_pose)
        for param_group in optimizer_pose.param_groups:
            param_group['lr'] = new_lrate_pose

        decay_rate_transform = args.decay_rate_transform
        new_lrate_trans = args.transform_lrate * (decay_rate_transform ** (global_step / decay_steps))
        logger.write("lr_trans", new_lrate_trans)
        for param_group in optimizer_trans.param_groups:
            param_group['lr'] = new_lrate_trans
        ###############################

        # print result in console
        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {loss.item()}, event_loss: {event_loss.item()}, rgb_loss: {rgb_loss.item()}, "
                f"event_fine_loss: {img_loss.item()}, event_coarse_loss: {img_loss0.item()}, "
                f"rgb_loss_fine: {rgb_loss_fine.item()}, rgb_loss_coarse: {rgb_loss_coarse.item()}")

        # save checkpoint
        if i % args.i_weights == 0 and i > 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'graph': graph.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_pose': optimizer_pose.state_dict(),
                'optimizer_trans': optimizer_trans.state_dict(),
            }, path)
            # save to logger
            logger.write_checkpoint(path, expname)
            print('Saved checkpoints at', path)

        # test
        if i % args.i_img == 0 and i > 0:
            with torch.no_grad():
                # imgs_render = render_image_test(i, graph, poses[0][:, :4].reshape(1, 3, 4), H, W, K, args,
                #                                 dir='test_poses_mid',
                #                                 need_depth=False)
                imgs, depth = render_image_test(i, graph, test_poses, H, W, K, args,
                                                dir='test', need_depth=False)
                if len(imgs) > 0:
                    logger.write_img("test_img_mid", imgs[len(imgs) // 2])
                    logger.write_imgs("test_img_all", imgs)
                    img_mid = imgs[len(imgs) // 2] / 255.
                    img_mid = torch.unsqueeze(torch.tensor(img_mid, dtype=torch.float32), dim=0)
                    test_mid_mse = compute_img_metric(img_mid, imgtests, metric="mse")
                    test_mid_psnr = compute_img_metric(img_mid, imgtests, metric="psnr")
                    test_mid_ssim = compute_img_metric(img_mid, imgtests, metric="ssim")
                    test_mid_lpips = compute_img_metric(img_mid, imgtests, metric="lpips")

                    logger.write("test_mid_mse", test_mid_mse)
                    logger.write("test_mid_psnr", test_mid_psnr)
                    logger.write("test_mid_ssim", test_mid_ssim)
                    logger.write("test_mid_lpips", test_mid_lpips)
                if len(depth) > 0:
                    logger.write_img("test_depth_mid", depth[len(depth) // 2])
                    logger.write_imgs("test_depth_all", depth)

        # if i % args.i_video == 0 and i > 0:
        #     bds = np.array([1 / 0.75, 150 / 0.75])
        #     optimized_se3 = graph.rgb_pose.end.weight.data
        #     optimized_pose = se3_to_SE3_N(optimized_se3)
        #     optimized_pose = torch.cat(
        #         [optimized_pose, torch.tensor([H, W, focal]).reshape([1, 3, 1]).repeat(optimized_pose.shape[0], 1, 1)],
        #         -1)
        #     optimized_pose = optimized_pose.cpu().numpy()
        #     render_poses = regenerate_pose(optimized_pose, bds, recenter=True, bd_factor=.75, spherify=False,
        #                                    path_zflat=False)
        #     # Turn on testing mode
        #     with torch.no_grad():  # here render_video有点问题
        #         rgbs, disps = render_video_test(i, graph, render_poses, H, W, K, args)
        #     print('Done, saving', rgbs.shape, disps.shape)
        #     moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
        #     imageio.mimwrite(moviebase + 'rgb.mp4', imgutils.to8bit(rgbs), fps=30, quality=8)
        #     imageio.mimwrite(moviebase + 'disp.mp4', imgutils.to8bit(disps / np.max(disps)), fps=30, quality=8)

        logger.update_buffer()
        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.random.manual_seed(0)
    random.seed(0)

    # load config
    print("Loading config")
    parser = config_parser()
    args = parser.parse_args()

    # setup device
    print(f"Use device: {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # train
    print("Start training...")
    train(args=args)
