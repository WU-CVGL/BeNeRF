import cv2
import os
import numpy as np
import imageio

NUM_I_RGB = 20
INTERVAL = 500
length = NUM_I_RGB * INTERVAL
NUM_IMG = NUM_I_RGB * INTERVAL + 1

START = 0

data_type = 'train'

basedir = r'D:\blender\scripts\BlenderCV\tanabata_1'
imgdir = os.path.join(basedir, 'camera/temp')
savedir_RGB = os.path.join(basedir, data_type + '_RGB')
savedir_RGB_start = os.path.join(basedir, data_type + '_RGB_start')
savedir_RGB_end = os.path.join(basedir, data_type + '_RGB_end')
savedir_Gray = os.path.join(basedir, data_type + '_Gray')
savedir_Gray_start = os.path.join(basedir, data_type + '_Gray_start')
savedir_Gray_end = os.path.join(basedir, data_type + '_Gray_end')
savedir_blur_RGB = os.path.join(basedir, data_type + '_blur_RGB')
savedir_blur_Gray = os.path.join(basedir, data_type + '_blur_Gray')

imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
imgfiles = [f for f in imgfiles if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]

imgfiles = imgfiles[START:START + NUM_IMG]


def imread(f):
    if f.endswith('png'):
        return imageio.v3.imread(f, ignoregamma=True)
    else:
        return imageio.v3.imread(f)


if __name__ == '__main__':
    os.makedirs(savedir_RGB, exist_ok=True)
    os.makedirs(savedir_Gray, exist_ok=True)
    os.makedirs(savedir_blur_RGB, exist_ok=True)
    os.makedirs(savedir_blur_Gray, exist_ok=True)
    os.makedirs(savedir_Gray_end, exist_ok=True)
    os.makedirs(savedir_Gray_start, exist_ok=True)
    os.makedirs(savedir_RGB_end, exist_ok=True)
    os.makedirs(savedir_RGB_start, exist_ok=True)

    img_id = INTERVAL // 2
    for i in range(NUM_I_RGB):
        RGB_dir = imgfiles[img_id]
        I_Gray = cv2.imread(RGB_dir, cv2.IMREAD_GRAYSCALE)
        dir = os.path.join(savedir_Gray, '{:06d}.png'.format(i))
        imageio.imwrite(dir, I_Gray)

        temp = []
        for temp_id in range(i * INTERVAL, (i + 1) * INTERVAL + 1):
            temp_dir = imgfiles[temp_id]
            temp_img = cv2.imread(temp_dir, cv2.IMREAD_GRAYSCALE)
            temp.append(temp_img)
        temp = np.stack(temp).astype(np.float32)
        temp = temp.mean(axis=0)
        temp = temp.astype(np.uint8)
        dir = os.path.join(savedir_blur_Gray, '{:06d}.png'.format(i))
        imageio.imwrite(dir, temp)

        img_id += INTERVAL

    img_id = INTERVAL // 2
    for i in range(NUM_I_RGB):
        RGB_dir = imgfiles[img_id]
        I_RGB = imageio.v3.imread(RGB_dir)
        dir = os.path.join(savedir_RGB, '{:06d}.png'.format(i))
        imageio.imwrite(dir, I_RGB)

        temp = []
        for temp_id in range(i * INTERVAL, (i + 1) * INTERVAL + 1):
            temp_dir = imgfiles[temp_id]
            temp_img = imageio.v3.imread(temp_dir)
            temp.append(temp_img)
        temp = np.stack(temp).astype(np.float32)
        temp = temp.mean(axis=0)
        temp = temp.astype(np.uint8)
        dir = os.path.join(savedir_blur_RGB, '{:06d}.png'.format(i))
        imageio.imwrite(dir, temp)

        img_id += INTERVAL

    img_id = 0
    for i in range(NUM_I_RGB):
        RGB_dir_start = imgfiles[img_id]
        I_RGB_start = imageio.v3.imread(RGB_dir_start)
        dir = os.path.join(savedir_RGB_start, '{:06d}.png'.format(i))
        imageio.imwrite(dir, I_RGB_start)
        img_id += INTERVAL

    img_id = INTERVAL
    for i in range(NUM_I_RGB - 1):
        RGB_dir_end = imgfiles[img_id]
        I_RGB = imageio.v3.imread(RGB_dir_end)
        dir = os.path.join(savedir_RGB_end, '{:06d}.png'.format(i))
        imageio.imwrite(dir, I_RGB)
        img_id += INTERVAL

    img_id = 0
    for i in range(NUM_I_RGB):
        Gray_dir_start = imgfiles[img_id]
        I_Gray_start = cv2.imread(Gray_dir_start, cv2.IMREAD_GRAYSCALE)
        dir = os.path.join(savedir_Gray_start, '{:06d}.png'.format(i))
        imageio.imwrite(dir, I_Gray_start)
        img_id += INTERVAL

    img_id = INTERVAL
    for i in range(NUM_I_RGB - 1):
        Gray_dir_end = imgfiles[img_id]
        I_Gray_end = cv2.imread(Gray_dir_end, cv2.IMREAD_GRAYSCALE)
        dir = os.path.join(savedir_Gray_end, '{:06d}.png'.format(i))
        imageio.imwrite(dir, I_Gray_end)
        img_id += INTERVAL

    img0 = cv2.imread(imgfiles[0], cv2.IMREAD_GRAYSCALE)

    [H, W] = img0.shape

    print('end!')
