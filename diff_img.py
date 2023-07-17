import imageio
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

img1_path = './data/rollingshutter-dataset/linear/train/start'
img2_path = './logs/linear/linear_only_optimize_SE3/test_poses_start/img_test_180000'
save_path = './data/rollingshutter-dataset/diff_res_start_test'


def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)


def load_imgs(path):
    imgfiles = [os.path.join(path, f) for f in sorted(os.listdir(path)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    imgs = imgs.astype(np.float32)

    return imgs


# def img_diff(img1_path, img2_path, save_path):
#     imgs1 = load_imgs(img1_path)
#     imgs2 = load_imgs(img2_path)
#     imgs_diff = np.mean((imgs1 - imgs2)**2, -1) * 20
#     os.makedirs(save_path, exist_ok=True)
#     for i in range(imgs_diff.shape[0]):
#         img_plot = plt.imshow(imgs_diff[i], cmap='gray')
#         plt.savefig(os.path.join(save_path, 'diff_{:03d}.png'.format(i)))

def img_diff(imgs1, imgs2, save_path, method=''):
    # imgs1 = load_imgs(img1_path)
    # imgs2 = load_imgs(img2_path)
    imgs_diff = np.array(np.abs((imgs2 - imgs1) * 256), dtype='uint8')
    # imgs_diff = np.array(np.abs((imgs2 - imgs1) * 256), dtype='uint8')
    # imgs_diff_1 = np.array((imgs2 - imgs1) * 128 + np.abs((imgs2 - imgs1) * 128), dtype='uint8')
    # imgs_diff_2 = np.array((imgs1 - imgs2) * 128 + np.abs((imgs1 - imgs2) * 128), dtype='uint8')
    os.makedirs(save_path, exist_ok=True)
    for i in range(imgs_diff.shape[0]):
        imageio.imwrite(os.path.join(save_path, method + 'Res_image_{:03d}.png'.format(i)), imgs_diff[i])
        # imageio.imwrite(os.path.join(save_path, '[image2-image1]_{:03d}.png'.format(i)), imgs_diff_1[i])
        # imageio.imwrite(os.path.join(save_path, '[image1-image2]_{:03d}.png'.format(i)), imgs_diff_2[i])


if __name__ == '__main__':
    # img1_path = os.path.join(img1_path, '000.png')
    # img2_path = os.path.join(img2_path, '001.png')
    img_diff(img1_path, img2_path, save_path)
