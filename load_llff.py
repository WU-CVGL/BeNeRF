import os

import numpy as np
import torch
from imageio.v3 import imread

from downsample import downsample


########## Slightly modified version of LLFF data loading code
##########  see https://github.com/Fyusion/LLFF for original

# downsample
def _minify(basedir, factors=[], resolutions=[]):  # basedir: ./data/nerf_llff_data/fern
    needtoload = False
    for r in factors:  # factors?
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:  # resolutions?
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.lower().endswith(ex) for ex in ['jpg', 'png', 'jpeg']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir, factor=None, width=None, height=None, load_pose=False, gray=False):
    poses_ts = np.loadtxt(os.path.join(basedir, 'poses_ts.txt'))

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        if not os.path.exists(os.path.join(basedir, 'images' + sfx)):
            downsample(basedir, factor, 'images')

        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    testdir = os.path.join(basedir, 'images' + "_test")
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    imgtests = [os.path.join(testdir, f) for f in sorted(os.listdir(testdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    if gray:
        imgs = [imread(f) / 255. for f in imgfiles]
    else:
        imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    if gray:
        imgtests = [imread(f) / 255. for f in imgtests]
    else:
        imgtests = [imread(f)[..., :3] / 255. for f in imgtests]
    imgtests = np.stack(imgtests, -1)

    # load poses
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # 3x5xN
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)  # 列的转换    -y x z : x y z
    # poses: [3, 5, N]->[N, 3, 5] imgs: [HWCN]->[NHWC] bds: [2, N]->[N, 2]
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)

    ev_poses_arr = np.load(os.path.join(basedir, 'poses_bounds_events.npy'))
    ev_poses = ev_poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    ev_poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    ev_poses = np.concatenate([ev_poses[:, 1:2, :], -ev_poses[:, 0:1, :], ev_poses[:, 2:, :]], 1)
    ev_poses = np.moveaxis(ev_poses, -1, 0).astype(np.float32)
    print('Loaded image data', imgs.shape)

    return imgs, imgtests, poses_ts, poses, ev_poses



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


def load_llff_data(basedir, factor=1, idx=0, deblur_dataset=50, gray=False, load_pose=False):
    imgs, imgtests, poses_ts, poses, ev_poses = _load_data(basedir, factor=factor, load_pose=load_pose, gray=gray)

    print('Loaded', basedir)

    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    if gray:
        imgs = np.expand_dims(imgs, -1)
    imgs = np.expand_dims(imgs[idx], 0)
    imgs = torch.Tensor(imgs)

    imgtests = np.moveaxis(imgtests, -1, 0).astype(np.float32)
    if gray:
        imgtests = np.expand_dims(imgtests, -1)
    imgtests = np.expand_dims(imgtests[idx], 0)
    imgtests = torch.Tensor(imgtests)

    poses_ts = poses_ts[idx:idx + 2]

    eventdir = os.path.join(basedir, "events")
    if os.path.exists(os.path.join(eventdir, "events.npy")):
        events = np.load(os.path.join(eventdir, "events.npy"))
        delta = (poses_ts[1] - poses_ts[0]) * 0.001
        events = np.array([event for event in events if poses_ts[0] - delta <= event[2] <= poses_ts[1] + delta])
    else:
        eventfiles = [os.path.join(eventdir, f) for f in sorted(os.listdir(eventdir)) if
                      f.endswith('npy') and f.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))]
        eventfiles = eventfiles[deblur_dataset * idx: deblur_dataset * (idx + 1)]

        event_list = [np.load(e) for e in eventfiles]
        events = np.concatenate(event_list)
        events = events[events[:, 2].argsort()]

    # create dictionary
    events = {'x': events[:, 0].astype(int), 'y': events[:, 1].astype(int), 'ts': events[:, 2], 'pol': events[:, 3],
              'num': events.shape[0]}
    if load_pose:
        size_rgb = poses.shape[0]
        # recenter for rgb
        poses_all = np.concatenate((poses, ev_poses), axis=0)
        poses_all = recenter_poses(poses_all)
        poses = poses_all[idx: idx + 2]

        # recenter for event
        ev_poses = poses_all[size_rgb + idx: size_rgb + idx + 2]

        return events, imgs, imgtests, poses_ts, poses, ev_poses

    return events, imgs, imgtests, poses_ts


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
