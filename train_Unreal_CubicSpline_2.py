import os
import random

import imageio
import torch.nn
from tqdm import trange, tqdm

import optimize_pose_CubicSpline_2
from config import config_parser
from logger.wandb_logger import WandbLogger
from loss import imgloss
from spline import se3_to_SE3_N, SE3_to_se3_N
from load_llff import regenerate_pose, load_llff_data
from nerf import *
from run_nerf_helpers import render_video_test, to8b, init_nerf, mse2psnr, render_image_test
from utils import imgutils
from utils.mathutils import safelog


def train(args):
    print('args.barf: ', args.barf, 'args.barf_start_end: ', args.barf_start, args.barf_end)

    # Load data images are groundtruth
    logger = WandbLogger(args)
    load_state = args.load_state

    # transforms
    mse_loss = imgloss.MSELoss()
    rgb2gray = imgutils.RGB2Gray()

    K = None

    if args.dataset_type == 'llff':
        events, images, poses, bds_start, render_poses, poses_ts = load_llff_data(args.datadir, args.threshold,
                                                                                  pose_state=load_state,
                                                                                  factor=args.factor, recenter=True,
                                                                                  bd_factor=.75, spherify=args.spherify,
                                                                                  focal=args.focal)

        poses = poses.repeat(args.trajectory_seg_num, 1, 1)

        hwf = poses[0, :3, -1]

        # split train/val/test
        i_test = torch.tensor([poses.shape[0], poses.shape[0] + 1, poses.shape[0] + 2]).long()
        i_val = i_test
        # i_train = torch.Tensor([i for i in torch.arange(int(images.shape[0])) if
        #                         (i not in i_test and i not in i_val)]).long()

        i_train = torch.arange(2)

        i_train = (i_train.reshape(-1, 1) + i_train.shape[0] * torch.arange(args.trajectory_seg_num)).reshape(-1)
        # poses_start_se3 = SE3_to_se3_N(poses_start[:, :3, :4])[i_train]    # pose 转换为 旋转角速度+运动速度
        poses_se3 = SE3_to_se3_N(poses[:, :3, :4])[i_train]

        print('Loaded data', images.shape, render_poses.shape, hwf, args.datadir)

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = torch.min(bds_start) * .9
            far = torch.max(bds_start) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)

    if K is None:
        K = torch.Tensor([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    print('camera intrinsic parameters: ', K, ' !!!')

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    test_metric_file = os.path.join(basedir, expname, 'test_metrics.txt')
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

    ### fixme !!! test nerf.py
    '''
    model = Model()
    graph = model.build_network(args)  # nerf, nerf_fine, forward
    optimizer = model.setup_optimizer(args)
    '''
    if args.load_weights:
        low, high = 0.0001, 0.001
        rand = (high - low) * torch.rand(poses_se3.shape[0], 6) + low
        poses_se3 = poses_se3 + rand

        model = optimize_pose_CubicSpline_2.Model(poses_se3, poses_ts)  # 22和25
        graph = model.build_network(args)
        optimizer, optimizer_pose, optimizer_trans = model.setup_optimizer(args)
        path = os.path.join(basedir, expname, '{:06d}.tar'.format(args.weight_iter))  # here
        graph_ckpt = torch.load(path)

        if args.only_optimize_SE3:
            # path_ = os.path.join(basedir, expname, '{:06d}.tar'.format(180000))  # here
            # graph_ckpt_ = torch.load(path_)

            # only load nerf and nerf_fine network
            delete_key = []
            for key, value in graph_ckpt['graph'].items():
                if key[:4] == 'se3.':
                    delete_key.append(key)

            pretrained_dict = {k: v for k, v in graph_ckpt['graph'].items() if k not in delete_key}
            print('only load nerf and nerf_fine network!!!!!')
            graph.load_state_dict(pretrained_dict, strict=False)
            global_step = 1
        else:
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
        if args.optimize_se3:
            low, high = 0.0001, 0.001
            rand = (high - low) * torch.rand(poses_se3.shape[0], 6) + low
            poses_se3 = poses_se3 + rand
        model = optimize_pose_CubicSpline_2.Model(poses_se3, poses_ts)
        graph = model.build_network(args)  # nerf, nerf_fine, forward
        optimizer, optimizer_pose, optimizer_trans = model.setup_optimizer(args)
        print('Not Load Model!')

    N_iters = args.max_iter + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = 0
    if not args.load_weights:
        global_step = start
    global_step_ = global_step
    threshold = N_iters + 101

    # show image
    num = images.shape[0]

    print('Number is {} !!!'.format(num))
    for i in trange(start, N_iters):
        i = i + global_step_
        if i == 0:
            init_nerf(graph.nerf)
            init_nerf(graph.nerf_fine)

        if i % args.i_video == 0 and i > 0:
            ret, ray_idx, test_poses, events_accu = graph.forward(i, poses_ts, threshold, events,
                                                                  H, W, K, args)

        elif i % args.i_img == 0 and i > 0:
            ret, ray_idx, test_poses, events_accu = graph.forward(i, poses_ts, threshold, events,
                                                                  H, W, K, args)

        else:
            ret, ray_idx, events_accu = graph.forward(i, poses_ts, threshold, events, H, W, K,
                                                      args)

        pixels_num = ray_idx.shape[0]

        ret_gray1 = {'rgb_map': ret['rgb_map'][:pixels_num],
                     'rgb0': ret['rgb0'][:pixels_num]}
        ret_gray2 = {'rgb_map': ret['rgb_map'][pixels_num:pixels_num * 2],
                     'rgb0': ret['rgb0'][pixels_num:pixels_num * 2]}
        ret_rgb = {'rgb_map': ret['rgb_map'][pixels_num * 2:],
                   'rgb0': ret['rgb0'][pixels_num * 2:]}

        target_s = events_accu.reshape(-1, 1)[ray_idx]

        # backward
        optimizer_pose.zero_grad()
        optimizer_trans.zero_grad()
        optimizer.zero_grad()

        img_loss = mse_loss(safelog(rgb2gray(ret_gray2['rgb_map'])) - safelog(rgb2gray(ret_gray1['rgb_map'])), target_s)
        psnr = mse2psnr(img_loss)
        img_loss *= args.event_coefficient
        logger.write("train_event_loss_fine", img_loss.item())

        # Event loss
        if 'rgb0' in ret:
            img_loss0 = mse_loss(safelog(rgb2gray(ret_gray2['rgb0'])) - safelog(rgb2gray(ret_gray1['rgb0'])), target_s)
            psnr0 = mse2psnr(img_loss0)
            img_loss0 *= args.event_coefficient
            logger.write("train_event_loss_coarse", img_loss0.item())


        event_loss = img_loss0 + img_loss

        loss = event_loss

        # RGB loss
        if args.rgb_loss:
            target_s = images[0].reshape(-1, H * W, 3)
            target_s = target_s[:, ray_idx]
            target_s = target_s.reshape(-1, 3)
            interval = target_s.shape[0]
            rgb_ = 0
            extras_ = 0
            rgb_list = []
            extras_list = []
            for j in range(0, args.deblur_images):
                rgb_ += ret_rgb['rgb_map'][j * interval:(j + 1) * interval]
                if 'rgb0' in ret_rgb:
                    extras_ += ret_rgb['rgb0'][j * interval:(j + 1) * interval]
                if (j + 1) % args.deblur_images == 0:
                    rgb_ = rgb_ / args.deblur_images
                    rgb_list.append(rgb_)
                    rgb_ = 0
                    if 'rgb0' in ret_rgb:
                        extras_ = extras_ / args.deblur_images
                        extras_list.append(extras_)
                        extras_ = 0

            rgb_blur = torch.stack(rgb_list, 0)
            rgb_blur = rgb_blur.reshape(-1, 3)

            if 'rgb0' in ret:
                extras_blur = torch.stack(extras_list, 0)
                extras_blur = extras_blur.reshape(-1, 3)

            rgb_loss_fine = mse_loss(rgb_blur, target_s)
            rgb_loss_fine *= args.rgb_coefficient
            logger.write("train_rgb_loss_fine", rgb_loss_fine.item())

            if 'rgb0' in ret:
                rgb_loss_coarse = mse_loss(extras_blur, target_s)
                rgb_loss_coarse *= args.rgb_coefficient
                logger.write("train_rgb_loss_coarse", rgb_loss_coarse.item())

            rgb_loss = rgb_loss_fine + rgb_loss_coarse
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
            optimizer.step()
        if args.optimize_se3:
            optimizer_pose.step()
        if args.optimize_trans:
            optimizer_trans.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = args.decay_rate
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        decay_rate_pose = args.decay_rate_pose
        new_lrate_pose = args.pose_lrate * (decay_rate_pose ** (global_step / decay_steps))
        for param_group in optimizer_pose.param_groups:
            param_group['lr'] = new_lrate_pose

        decay_rate_transform = args.decay_rate_transform
        new_lrate_trans = args.transform_lrate * (decay_rate_transform ** (global_step / decay_steps))
        for param_group in optimizer_trans.param_groups:
            param_group['lr'] = new_lrate_trans
        ###############################

        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {loss.item()}, event_loss: {event_loss.item()}, rgb_loss: {rgb_loss.item()}, "
                f"event_fine_loss: {img_loss.item()}, event_coarse_loss: {img_loss0.item()}, "
                f"rgb_loss_fine: {rgb_loss_fine.item()}, rgb_loss_coarse: {rgb_loss_coarse.item()}")

        if i < 20:
            print(
                f"[TRAIN] Iter: {i} Loss: {loss.item()}, event_loss: {event_loss.item()}, rgb_loss: {rgb_loss.item()}\n"
                f"event_fine_loss: {img_loss.item()}, event_coarse_loss: {img_loss0.item()}, \n"
                f"rgb_loss_fine: {rgb_loss_fine.item()}, rgb_loss_coarse: {rgb_loss_coarse.item()}\n")

        if i % args.i_weights == 0 and i > 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'graph': graph.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_pose': optimizer_pose.state_dict(),
                'optimizer_trans': optimizer_trans.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_img == 0 and i > 0:
            with torch.no_grad():
                # imgs_render = render_image_test(i, graph, poses[0][:, :4].reshape(1, 3, 4), H, W, K, args,
                #                                 dir='test_poses_mid',
                #                                 need_depth=False)
                imgs_render = render_image_test(i, graph, test_poses, H, W, K, args,
                                                dir='test_poses_mid',
                                                need_depth=False)

        if i % args.i_video == 0 and i > 0:
            bds = np.array([1 / 0.75, 150 / 0.75])
            optimized_se3 = graph.rgb_pose.end.weight.data
            optimized_pose = se3_to_SE3_N(optimized_se3)
            optimized_pose = torch.cat(
                [optimized_pose, torch.tensor([H, W, focal]).reshape([1, 3, 1]).repeat(optimized_pose.shape[0], 1, 1)],
                -1)
            optimized_pose = optimized_pose.cpu().numpy()
            render_poses = regenerate_pose(optimized_pose, bds, recenter=True, bd_factor=.75, spherify=False,
                                           path_zflat=False)
            # Turn on testing mode
            with torch.no_grad():  # here render_video有点问题
                rgbs, disps = render_video_test(i, graph, render_poses, H, W, K, args)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        logger.update_buffer()
        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.random.manual_seed(0)
    random.seed(0)

    # load config
    parser = config_parser()
    args = parser.parse_args()

    # setup device
    torch.cuda.set_device(args.device)

    # train
    print('Cubic Spline 2!!!\n')
    train(args=args)
