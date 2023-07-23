import os

import numpy as np
import torch
from imageio.v3 import imread


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


def _load_data(basedir, factor=None):
    poses_ts = np.loadtxt(os.path.join(basedir, 'poses_ts.txt'))

    imgdir = os.path.join(basedir, 'images_blur')
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    imgs = [imread(f) / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    testdir = os.path.join(basedir, 'images_test')
    if not os.path.exists(testdir):
        print(testdir, 'does not exist, returning')
        return

    imgtests = [os.path.join(testdir, f) for f in sorted(os.listdir(testdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    imgtests = [imread(f) / 255. for f in imgtests]
    imgtests = np.stack(imgtests, -1)

    events = np.loadtxt(os.path.join(basedir, 'events.txt'))

    print('Loaded image data', imgs.shape)
    return imgs, imgtests, events, poses_ts

def load_davis_data(basedir, factor=1, idx=0):
    imgs, imgtests, events, poses_ts = _load_data(basedir, factor=factor)
    print('Loaded', basedir)

    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    imgs = np.expand_dims(imgs, -1)
    imgs = np.expand_dims(imgs[idx], 0)
    imgs = torch.Tensor(imgs)

    imgtests = np.moveaxis(imgtests, -1, 0).astype(np.float32)
    imgtests = np.expand_dims(imgtests, -1)
    imgtests = np.expand_dims(imgtests[idx], 0)
    imgtests = torch.Tensor(imgtests)

    poses_ts = poses_ts[idx:idx + 2]
    events = np.array([event for event in events if poses_ts[0] <= event[0] <= poses_ts[1]])

    # create dictionary
    events = {'x': events[:, 1].astype(int), 'y': events[:, 2].astype(int), 'ts': events[:, 0], 'pol': events[:, 3],
              'num': events.shape[0]}

    return events, imgs, imgtests, poses_ts
