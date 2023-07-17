import os

import torch

from nerf_chunk import *
import optimize_pose_CubicSpline_2_chunk
import compose
import torchvision.transforms.functional as torchvision_F

import matplotlib.pyplot as plt

from downsample import downsample

from tvloss import EdgeAwareVariationLoss, GrayEdgeAwareVariationLoss
from metrics import compute_img_metric

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.set_device(1)

log_eps = 1e-3
r = 0.299
g = 0.587
b = 0.114
rgb_weight = torch.Tensor([r, g, b]).to(device)
RGB_2_Gray = lambda x: torch.sum(x * rgb_weight[None, :], axis=-1)
log = lambda x: torch.log(x + log_eps)


def train(args):
    print('args.barf: ', args.barf, 'args.barf_start_end: ', args.barf_start, args.barf_end)
    print('tv_width: ', args.tv_width_nerf)
    print('lambda: ', args.tv_loss_lambda)

    # Load data images are groundtruth
    # use_GT = False
    optimize_se3 = args.optimize_se3
    optimize_nerf = args.optimize_nerf
    load_state = args.load_state
    # load_state = 'start_end'
    print('!!! Optimize SE3 Network: ', optimize_se3)
    print('!!! Load which npy file: ', load_state)
    print('load poses correspond to %s line' % args.load_state)
    # focal_GT = torch.tensor([548.409])

    K = None

    if args.dataset_type == 'llff':
        events, images, poses, bds_start, render_poses, poses_ts = load_llff_data(args.datadir, args.threshold,
                                                                                  pose_state=load_state,
                                                                                  factor=args.factor, recenter=True,
                                                                                  bd_factor=.75, spherify=args.spherify,
                                                                                  focal=args.focal)

        if images.shape[0] != poses.shape[0]:  # 代表每一张图不止对应一个 pose
            if (images.shape[0] - 1) * (args.trajectory_seg_num + 1) == poses.shape[0] and args.load_state == 'mid':
                pos = images.shape[0] * args.trajectory_seg_num  # first N: start; last one: end
                poses_end = poses_start[pos:]
                poses_start = poses_start[:pos]
                print('!!! Initialize the SE3 Network with GT of start pose and end pose, the number of poses is ',
                      poses_start.shape[0])
            elif images.shape[0] == poses.shape[0] + 1 and args.load_state == 'mid':
                poses = poses  # initialize with poses corresponding to middle lines
                print('!!! Initializzation with poses corresponding to middle lines')
            else:
                print(
                    'Mismatch between the number of trajectory_seg {} and poses {} !!!!'.format(args.trajectory_seg_num,
                                                                                                poses_start.shape[0]))
                return

        else:
            poses = poses.repeat(args.trajectory_seg_num, 1, 1)
            print('!!! Initializzation with poses corresponding to whole trajectory')

        hwf = poses[0, :3, -1]

        # split train/val/test
        i_test = torch.tensor([poses.shape[0], poses.shape[0] + 1, poses.shape[0] + 2]).long()
        i_val = i_test
        i_train = torch.Tensor([i for i in torch.arange(int(images.shape[0])) if
                                (i not in i_test and i not in i_val)]).long()

        i_train = (i_train.reshape(-1, 1) + i_train.shape[0] * torch.arange(args.trajectory_seg_num)).reshape(-1)
        # poses_start_se3 = SE3_to_se3_N(poses_start[:, :3, :4])[i_train]    # pose 转换为 旋转角速度+运动速度
        poses_se3 = SE3_to_se3_N(poses[:, :3, :4])[i_train]

        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

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
    hwf = [H, W, focal]

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

        model = optimize_pose_CubicSpline_2_chunk.Model(poses_se3, poses_ts)  # 22和25
        graph = model.build_network(args, events)
        optimizer, optimizer_se3 = model.setup_optimizer(args)
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
            optimizer_se3.load_state_dict(graph_ckpt['optimizer_se3'])
            if args.two_phase:
                global_step = 1
            else:
                global_step = graph_ckpt['global_step']

        print('Model Load Done!')
    else:
        low, high = 0.0001, 0.001
        rand = (high - low) * torch.rand(poses_se3.shape[0], 6) + low
        if optimize_se3:  # 只有当需要优化 se3 时才证明目前load 的pose不够精准，否则加上 random noise 会使得性能下降一点点
            poses_se3 = poses_se3 + rand
        model = optimize_pose_CubicSpline_2_chunk.Model(poses_se3, poses_ts)
        graph = model.build_network(args, events)  # nerf, nerf_fine, forward
        optimizer, optimizer_se3 = model.setup_optimizer(args)
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

    if args.tv_loss:
        tvloss_fn_rgb = EdgeAwareVariationLoss(in1_nc=3)
        tvloss_fn_gray = GrayEdgeAwareVariationLoss(in1_nc=1, in2_nc=3)

    print('Number is {} !!!'.format(num))
    for i in trange(start, N_iters):
        ### core optimization loop ###
        # if i%500==0 and i>0:
        #     num += 1
        #     print('Number is {} !!!'.format(num))
        i = i + global_step_
        if i == 0:
            init_nerf(graph.nerf)
            init_nerf(graph.nerf_fine)

        if i % args.i_video == 0 and i > 0:
            ret, ray_idx, test_poses, events_accu, ret_rgb, ray_idx_rgb = graph.forward(i, poses_ts, threshold, events,
                                                                                        H, W, K, args)

        elif i % args.i_img == 0 and i > 0:
            ret, ray_idx, test_poses, events_accu, ret_rgb, ray_idx_rgb = graph.forward(i, poses_ts, threshold, events,
                                                                                        H, W, K, args)

        else:
            ret, ray_idx, events_accu, ret_rgb, ray_idx_rgb = graph.forward(i, poses_ts, threshold, events, H, W, K,
                                                                            args)

        pixels_num = ray_idx.shape[0]

        ret_Gray1 = {'rgb_map': ret['rgb_map'][:pixels_num], 'rgb0': ret['rgb0'][:pixels_num]}
        ret_Gray2 = {'rgb_map': ret['rgb_map'][pixels_num:], 'rgb0': ret['rgb0'][pixels_num:]}

        target_s = events_accu.reshape(-1, 1)[ray_idx]

        # target_s = target_s.reshape(-1, 3)

        if args.tv_loss and i >= args.n_tvloss:
            # if args.tv_loss_rgb:
            rgb_sharp_tv = ret['rgb_map'][-args.tv_width_nerf ** 2:]
            rgb_sharp_tv = rgb_sharp_tv.reshape(-1, args.tv_width_nerf, args.tv_width_nerf, 3)  # [NHWC]
            rgb_sharp_tv = torch.permute(rgb_sharp_tv, (0, 3, 1, 2))
            if args.tv_loss_gray:
                gray = ret['disp_map'][-args.tv_width_nerf ** 2:]
                depth_tv = gray
                depth_tv = depth_tv.reshape(-1, args.tv_width_nerf, args.tv_width_nerf, 1)  # [NHWC]
                depth_tv = torch.permute(depth_tv, (0, 3, 1, 2))

        # backward
        optimizer_se3.zero_grad()  # here
        optimizer.zero_grad()

        if args.tv_loss is False:
            img_loss = img2mse(log(ret_Gray2['rgb_map']) - log(ret_Gray1['rgb_map']), target_s)
        else:
            img_loss = img2mse(ret['rgb_map'][:-args.tv_width_nerf ** 2], target_s)

        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in ret:
            if args.tv_loss is False:
                img_loss0 = img2mse(log(ret_Gray2['rgb0']) - log(ret_Gray1['rgb0']), target_s)
            else:
                img_loss0 = img2mse(ret['rgb0'][:-args.tv_width_nerf ** 2], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        if args.tv_loss and i >= args.n_tvloss:
            if args.tv_loss_rgb and args.tv_loss_gray:
                loss_tv_rgb = tvloss_fn_rgb(rgb_sharp_tv, mean=True)
                loss_tv_depth = tvloss_fn_gray(depth_tv, rgb_sharp_tv, mean=True)
                loss_tv = loss_tv_rgb + loss_tv_depth
            elif args.tv_loss_rgb:
                loss_tv = tvloss_fn_rgb(rgb_sharp_tv, mean=True)
            else:
                loss_tv = tvloss_fn_gray(depth_tv, rgb_sharp_tv, mean=True)

            loss = loss + args.tv_loss_lambda * loss_tv

        if args.rgb_loss:
            target_s_rgb = images[0].reshape(H * W, 3)[ray_idx_rgb]
            rgb_loss = img2mse(ret_rgb['rgb_map'], target_s_rgb) + img2mse(ret_rgb['rgb0'], target_s_rgb)
            loss = loss + rgb_loss
        else:
            rgb_loss = torch.tensor(0)

        loss.backward()

        # if i<=threshold:
        if optimize_nerf:
            optimizer.step()
        if optimize_se3:
            optimizer_se3.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = args.decay_rate
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        decay_rate_pose = args.decay_rate_pose
        new_lrate_pose = args.pose_lrate * (decay_rate_pose ** (global_step / decay_steps))
        for param_group in optimizer_se3.param_groups:
            param_group['lr'] = new_lrate_pose
        ###############################

        if i % args.i_print == 0:
            if args.tv_loss and i >= args.n_tvloss:
                tqdm.write(
                    f"[TRAIN] Iter: {i} Loss: {loss.item()}  coarse_loss:, {img_loss0.item()}, rgb_loss: {rgb_loss.item()} TV Loss: {loss_tv.item()}")
            else:
                tqdm.write(
                    f"[TRAIN] Iter: {i} Loss: {loss.item()}  coarse_loss:, {img_loss0.item()}, rgb_loss: {rgb_loss.item()}")

        if i < 10:
            print('coarse_loss:', img_loss0)

        if i % args.i_weights == 0 and i > 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'graph': graph.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_se3': optimizer_se3.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_img == 0 and i > 0:
            with torch.no_grad():
                imgs_render = render_image_test(i, graph, test_poses, H, W, K, args, dir='test_poses_mid',
                                                need_depth=False)

        if i % args.i_video == 0 and i > 0:
            bds = np.array([1 / 0.75, 150 / 0.75])
            optimized_se3 = graph.se3.end.weight.data
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

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    print('Cubic Spline 2!!!\n')
    train(args=args)
