import numpy as np
import os
import imageio

to8b = lambda x: x.astype(np.uint8)


def downsample(imgdir, factor, folder):
    sfx = '_{}'.format(factor)
    savedir = os.path.join(imgdir, folder + sfx)

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    imgdir_1 = os.path.join(imgdir, folder)

    imgfiles = [os.path.join(imgdir_1, f) for f in sorted(os.listdir(imgdir_1)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('bmp')]

    def imread(f):
        return imageio.v3.imread(f)

    imgs = [imread(f)[..., :3] for f in imgfiles]
    imgs = np.stack(imgs, 0)
    sh = imgs.shape
    sh = np.array(sh)
    sh[1:3] = sh[1:3] / factor
    # imgs = np.stack(imgs, -1)
    x_array = np.arange(0, sh[2] * factor, factor).tolist()
    y_array = np.arange(0, sh[1] * factor, factor).tolist()

    new_imgs = imgs[:, y_array, :, :][:, :, x_array, :]

    for i in range(sh[0]):
        if savedir is not None:
            rgb8 = to8b(new_imgs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
