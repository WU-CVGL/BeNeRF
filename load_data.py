import os

import numpy as np
import torch

from utils import imgutils


def load_img_data(datadir, gray=False):
    print("Loading images...")
    # Load images
    imgdir = os.path.join(datadir, 'images')
    testdir = os.path.join(datadir, 'images' + "_test")

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.lower().endswith(("jpg", "png"))]
    imgtests = [os.path.join(testdir, f) for f in sorted(os.listdir(testdir)) if f.lower().endswith(("jpg", "png"))]

    imgs = [imgutils.load_image(f, gray) for f in imgfiles]
    imgs = np.stack(imgs, -1)

    imgtests = [imgutils.load_image(f, gray) for f in imgtests]
    imgtests = np.stack(imgtests, -1)

    return imgs, imgtests


def load_camera_pose(basedir, H, W, cubic):
    sh = H, W
    # load poses
    if cubic:
        poses_arr = np.load(os.path.join(basedir, 'poses_bounds_cubic.npy'))
        ev_poses_arr = np.load(os.path.join(basedir, 'poses_bounds_cubic_events.npy'))
    else:
        poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
        ev_poses_arr = np.load(os.path.join(basedir, 'poses_bounds_events.npy'))

    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # 3x5xN
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)  # 列的转换    -y x z : x y z
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)

    ev_poses = ev_poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    ev_poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    ev_poses = np.concatenate([ev_poses[:, 1:2, :], -ev_poses[:, 0:1, :], ev_poses[:, 2:, :]], 1)
    ev_poses = np.moveaxis(ev_poses, -1, 0).astype(np.float32)

    return poses, ev_poses


def load_camera_trans(basedir):
    # load trans
    trans_arr = np.load(os.path.join(basedir, 'trans.npy'))
    return trans_arr


def load_timestamps(basedir):
    poses_ts_path = os.path.join(basedir, 'poses_ts.txt')
    poses_start_path = os.path.join(basedir, "poses_start_ts.txt")
    poses_end_path = os.path.join(basedir, "poses_end_ts.txt")
    if os.path.exists(poses_ts_path):
        poses_ts = np.loadtxt(poses_ts_path)
        poses_start = poses_ts[:-1]
        poses_end = poses_ts[1:]
    elif os.path.exists(poses_start_path) and os.path.exists(poses_end_path):
        poses_start = np.loadtxt(poses_start_path)
        poses_end = np.loadtxt(poses_end_path)
    else:
        print("Cannot load timestamps for images")
        assert False

    return poses_start, poses_end


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def spherify_poses(poses, bds):
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, new_poses, bds


def load_data(datadir, args, load_pose=False, load_trans=False, cubic=False):
    datadir = os.path.expanduser(datadir)
    gray = args.channels == 1

    # process imges
    imgs, imgtests = load_img_data(datadir, gray=gray)

    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    if gray:
        imgs = np.expand_dims(imgs, -1)
    imgs = np.expand_dims(imgs[args.idx], 0)
    imgs = torch.Tensor(imgs)

    imgtests = np.moveaxis(imgtests, -1, 0).astype(np.float32)
    if gray:
        imgtests = np.expand_dims(imgtests, -1)
    imgtests = np.expand_dims(imgtests[args.idx], 0)
    imgtests = torch.Tensor(imgtests)

    # process events and timestamps
    ts_start, ts_end = load_timestamps(datadir)
    eventdir = os.path.join(datadir, "events")
    if os.path.exists(os.path.join(eventdir, "events.npy")):
        # event shift, selecting more events means better result
        st = max(args.idx - args.event_shift_start, 0)
        ed = min(args.idx + args.event_shift_end, len(ts_end) - 1)
        poses_ts = np.array((ts_start[st], ts_end[ed]))
        events = np.load(os.path.join(eventdir, "events.npy"))
        delta = (poses_ts[1] - poses_ts[0]) * args.event_time_shift
        poses_ts = np.array([poses_ts[0] - delta, poses_ts[1] + delta])
        events = np.array([event for event in events if poses_ts[0] <= event[2] <= poses_ts[1]])
    else:
        # synthesis dataset
        poses_ts = np.array((ts_start[args.idx], ts_end[args.idx]))
        eventfiles = [os.path.join(eventdir, f) for f in sorted(os.listdir(eventdir)) if
                      f.endswith('npy') and f.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))]
        eventfiles = eventfiles[args.dataset_event_split * args.idx: args.dataset_event_split * (args.idx + 1)]

        event_list = [np.load(e) for e in eventfiles]
        events = np.concatenate(event_list)

    events = events[events[:, 2].argsort()]
    # create dictionary
    events = {'x': events[:, 0].astype(int), 'y': events[:, 1].astype(int),
              # norm ts
              'ts': (events[:, 2] - poses_ts[0]) / (poses_ts[1] - poses_ts[0]), 'pol': events[:, 3]}

    # process poses
    poses, ev_poses, trans = None, None, None
    if load_pose:
        poses, ev_poses = load_camera_pose(datadir, imgs.shape[0], imgs.shape[1], cubic)
        # recenter for rgb
        poses_num = 4 if cubic else 2
        poses_all = np.concatenate((poses[args.idx: args.idx + 2], ev_poses[args.idx: args.idx + 2]), axis=0)
        poses_all = recenter_poses(poses_all)
        poses = poses_all[0:poses_num]

        # recenter for event
        ev_poses = poses_all[poses_num:2 * poses_num]
    elif load_trans:
        trans_arr = load_camera_trans(datadir)
        # trans_arr = np.expand_dims(trans_arr, axis=0)
        # trans = recenter_poses(trans_arr)[0]
        trans = trans_arr.astype(np.float32)

    return events, imgs, imgtests, poses_ts, poses, ev_poses, trans


def regenerate_pose(poses, bds, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        c2w = poses_avg(poses)
        # print('recentered', c2w.shape)
        # print(c2w[:3, :4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
            #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

    render_poses = np.array(render_poses).astype(np.float32)

    render_poses = torch.Tensor(render_poses)

    return render_poses
