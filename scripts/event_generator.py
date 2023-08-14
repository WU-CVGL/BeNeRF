import cv2
import os
import numpy as np
import imageio

"""
    event_sim_config.Cp = 0.05;
    event_sim_config.Cm = 0.03;
    event_sim_config.sigma_Cp = 0;
    event_sim_config.sigma_Cm = 0;
    event_sim_config.use_log_image = true;
    event_sim_config.log_eps = 0.001;
"""
NUM_I_RGB = 30
INTERVAL = 500
threshold = 0.1
log_eps = 1e-3
length = NUM_I_RGB * INTERVAL
NUM_Img = NUM_I_RGB * INTERVAL + 1

START = 0

data_type = 'train'

basedir = 'D:\\\EXP_ORIGINAL\\LivingRoom'
imgdir = os.path.join(basedir, 'camera/temp')
savedir_RGB = os.path.join(basedir, data_type + '_RGB')
savedir_RGB_start = os.path.join(basedir, data_type + '_RGB_start')
savedir_RGB_end = os.path.join(basedir, data_type + '_RGB_end')
savedir_Gray = os.path.join(basedir, data_type + '_Gray')
savedir_Gray_start = os.path.join(basedir, data_type + '_Gray_start')
savedir_Gray_end = os.path.join(basedir, data_type + '_Gray_end')
savedir_blur_RGB = os.path.join(basedir, data_type + '_blur_RGB')
savedir_blur_Gray = os.path.join(basedir, data_type + '_blur_Gray')
eventdir = os.path.join(basedir, data_type + '_event_threshold_' + '{:.1f}'.format(threshold))

if not os.path.exists(eventdir):
    os.makedirs(eventdir)

if not os.path.exists(savedir_RGB):
    os.makedirs(savedir_RGB)

if not os.path.exists(savedir_Gray):
    os.makedirs(savedir_Gray)

if not os.path.exists(savedir_blur_RGB):
    os.makedirs(savedir_blur_RGB)

if not os.path.exists(savedir_blur_Gray):
    os.makedirs(savedir_blur_Gray)

if not os.path.exists(savedir_Gray_end):
    os.makedirs(savedir_Gray_end)

if not os.path.exists(savedir_Gray_start):
    os.makedirs(savedir_Gray_start)

if not os.path.exists(savedir_RGB_end):
    os.makedirs(savedir_RGB_end)

if not os.path.exists(savedir_RGB_start):
    os.makedirs(savedir_RGB_start)

imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
imgfiles = [f for f in imgfiles if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]

imgfiles = imgfiles[START:START + NUM_Img]


def imread(f):
    if f.endswith('png'):
        return imageio.v3.imread(f, ignoregamma=True)
    else:
        return imageio.v3.imread(f)


groundtruth = np.loadtxt(os.path.join(basedir, 'groundtruth.txt'))
time_stamp = groundtruth[..., 0][START:START + NUM_Img]


def Gray_Event_Simulate(imgfiles, W):
    """"""
    n_pix_row = W

    img0 = cv2.imread(imgfiles[0], cv2.IMREAD_GRAYSCALE).flatten() / 1.0
    img_base = img0

    last_time = time_stamp[0] * np.ones_like(img_base)

    spike_time = last_time

    for i in range(length):
        eve_gene = np.zeros([0, 4])
        current_time = time_stamp[i + 1] * np.ones_like(img_base)
        img_pre = img_base
        img_next = cv2.imread(imgfiles[i + 1], cv2.IMREAD_GRAYSCALE).flatten() / 1.0

        deltaL = np.log(img_next + log_eps) - np.log(img_pre + log_eps)

        # spike_time = last_time

        spike_nums = (deltaL / threshold).astype(int)  # eve 只保存 整数 带正负号

        # if np.abs(deltaL) >= threshold:
        POL = np.sign(deltaL)
        # 对应的 event accumulate
        spikes = deltaL / threshold  #

        spike_pos = np.where(spike_nums != 0)[0]  # find how many pixels trigger events and where

        # get event datas for every pixel
        for spike_id in range(spike_pos.size):
            pixel_id = spike_pos[spike_id]  # find the pixel_id for spike_id-th event
            spike_num = spike_nums[pixel_id]  # spiked times for this pixels with sign

            spike_num_ = np.abs(spike_num)

            spike_time_temp = (last_time[pixel_id] + (1 + np.arange(spike_num_)) * (
                    current_time[pixel_id] - last_time[pixel_id]) / spikes[pixel_id] * POL[pixel_id]).reshape(-1, 1)
            xs_temp = np.array([[pixel_id % n_pix_row]]).repeat(spike_num_, axis=0)
            ys_temp = np.array([[pixel_id // n_pix_row]]).repeat(spike_num_, axis=0)
            pol_temp = np.array([[POL[pixel_id]]]).repeat(spike_num_, axis=0)

            eve_temp = np.concatenate([xs_temp, ys_temp, spike_time_temp, pol_temp], 1)
            eve_gene = np.concatenate([eve_gene, eve_temp], 0)

            spike_time[pixel_id] = spike_time_temp[-1, 0]  # save the latest time spiking event

        last_time = current_time
        # save the image after generate the event, and this image will be the next base image, cuz the asyn- trigger
        img_base = (img_pre + log_eps) * np.exp(spike_nums * threshold) - log_eps
        # img_base = img_next

        print(i)

        eve_gene_sorted = eve_gene[eve_gene[:, 2].argsort(), :]

        event_file_name = os.path.join(eventdir, '{:06d}.txt'.format(i))

        # with open(event_file_name, "ab") as f:    # 
        with open(event_file_name, "wb") as f:
            np.savetxt(f, eve_gene_sorted, delimiter=" ", fmt='%.8f')

        event_file_name = os.path.join(eventdir, '{:06d}.npy'.format(i))
        np.save(event_file_name, eve_gene_sorted)
    """"""


if __name__ == '__main__':

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

    Gray_Event_Simulate(imgfiles, W)

    print('end!')
