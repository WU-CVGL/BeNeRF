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
NUM_I_RGB = 6
INTERVAL = 50
threshold = 0.1
log_eps = 1e-3
length = NUM_I_RGB * INTERVAL
NUM_Img = NUM_I_RGB * INTERVAL + 1

START = 950

basedir = '../Event-Datasets/Living_Room_1000Hz'
imgdir = os.path.join(basedir, 'camera/temp')
savedir_RGB = os.path.join(basedir, 'test_RGB')
savedir_Gray = os.path.join(basedir, 'test_Gray')
eventdir = os.path.join(basedir, 'test_event_threshold_' + '{:.1f}'.format(threshold))

if not os.path.exists(eventdir):
    os.makedirs(eventdir)

if not os.path.exists(savedir_RGB):
    os.makedirs(savedir_RGB)

if not os.path.exists(savedir_Gray):
    os.makedirs(savedir_Gray)

imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
imgfiles = [f for f in imgfiles if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]

# imgfiles = imgfiles[:NUM_Img]

imgfiles = imgfiles[START:START + NUM_Img]


def imread(f):
    if f.endswith('png'):
        return imageio.v3.imread(f, ignoregamma=True)
    else:
        return imageio.v3.imread(f)


# imgs = [imread(f)[..., :3]/1.0 for f in imgfiles]    # 保证数据类型都是 float 类型！！！
# imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE)/1.0 for f in imgfiles]

# imgs = np.stack(imgs, 0)

# img_id = 0

# for i in range(NUM_I_RGB+1):
#     I_RGB = np.array(imgs[img_id], dtype='uint8')
#     dir = os.path.join(savedir, '{:03d}.png'.format(i))
#     imageio.imwrite(dir, I_RGB)
#     img_id += INTERVAL

groundtruth = np.loadtxt(os.path.join(basedir, 'groundtruth.txt'))
time_stamp = groundtruth[..., 0][START:START + NUM_Img]


# imgs[..., 0] = imgs[..., 1] = imgs[..., 2] = (0.299 * imgs[..., 0] + 0.587 * imgs[..., 1] + 0.114 * imgs[..., 2])
# imgs = 0.5*imgs

# def Color_Event_Simulate(imgs):
#     time_stamp = np.array([0])
#     eve_gene = np.zeros([0, 4])
#     img_base = imgs[0]

#     for i in range(length):
#         img_pre = img_base
#         img_next = imgs[i+1]
#         sys_eve = np.log((img_next + log_eps)/(img_pre + log_eps))

#         img_temp = np.zeros_like(img_base).astype(float)    # zeros_like get uint8 type

#         eve_pos = np.where(np.abs(sys_eve) >= threshold)
#         # eve = (sys_eve[eve_pos] / threshold)
#         eve = (sys_eve[eve_pos]/threshold).astype(int)    # eve 只保存 整数

#         # 对应的 event accumulate
#         img_temp[eve_pos] = (sys_eve[eve_pos]/threshold).astype(int)    #
#         # img_temp[eve_pos] = (sys_eve[eve_pos] / threshold)

#         eve_temp = np.concatenate([eve_pos[0].reshape([-1, 1]), eve_pos[1].reshape([-1, 1]), eve_pos[2].reshape([-1, 1]), eve.reshape([-1, 1])], 1)
#         time_stamp = np.concatenate([time_stamp, np.array([time_stamp[i] + eve.size])], 0)

#         # save the image after generate the event, and this image will be the next base image, cuz the asyn- trigger
#         img_base = (img_pre + log_eps) * np.exp(img_temp * threshold) - log_eps

#         eve_gene = np.concatenate([eve_gene, eve_temp], 0)
#         # img_err = img_next - img_base
#         print(i)

#         np.save(os.path.join(imgdir, 'color_events_data.npy'), eve_gene)
#         np.save(os.path.join(imgdir, 'events_stamp.npy'), time_stamp)

# @njit(parallel=True)
# def Gray_Event_Simulate(imgs, W):
#     eve_gene = np.zeros([0, 4])
#     n_pix_row = W

#     for x in prange(imgs_gray_flat.shape[1]):
#         imgs_ = imgs[..., x]    # for specific position
#         img_base = imgs_[0]
#         last_time = time_stamp[0]

#         for i in range(length):
#             current_time = time_stamp[i+1]
#             img_pre = img_base
#             img_next = imgs_[i+1]
#             deltaL = np.log((img_next + log_eps)/(img_pre + log_eps))

#             spike_time = current_time

#             spike_nums = int(deltaL/threshold)    # eve 只保存 整数 带正负号

#             if np.abs(deltaL) >= threshold:
#                 POL = np.sign(deltaL)
#                 # 对应的 event accumulate
#                 spikes = deltaL/threshold    #

#                 for num in range(np.abs(spike_nums)):
#                     spike_time += (current_time - last_time) / spikes * POL
#                     eve_temp = np.array([x % n_pix_row, x // n_pix_row, spike_time, POL]).reshape(1,-1)
#                     eve_gene = np.concatenate([eve_gene, eve_temp], 0)

#             last_time = spike_time
#             # save the image after generate the event, and this image will be the next base image, cuz the asyn- trigger
#             img_base = (img_pre + log_eps) * np.exp(spike_nums * threshold) - log_eps
#             # img_err = img_next - img_base
#             # print(i)

#         print(x)

#     eve_gene_sorted = eve_gene[eve_gene[:,2].argsort(), :]

#     np.save(os.path.join(imgdir, 'gray_events_data.npy'), eve_gene_sorted)
#     np.savetxt(os.path.join(imgdir, 'gray_events_data.txt'), eve_gene_sorted, fmt = '%.8f')
# @numba.jit()
def Gray_Event_Simulate(imgfiles, W):
    """"""
    n_pix_row = W

    img0 = cv2.imread(imgfiles[0], cv2.IMREAD_GRAYSCALE).flatten() / 1.0
    img_base = img0

    last_time = time_stamp[0] * np.ones_like(img_base)

    for i in range(length):
        eve_gene = np.zeros([0, 4])
        current_time = time_stamp[i + 1] * np.ones_like(img_base)
        img_pre = img_base
        img_next = cv2.imread(imgfiles[i + 1], cv2.IMREAD_GRAYSCALE).flatten() / 1.0

        deltaL = np.log(img_next + log_eps) - np.log(img_pre + log_eps)

        spike_time = current_time

        spike_nums = (deltaL / threshold).astype(int)  # eve 只保存 整数 带正负号

        # if np.abs(deltaL) >= threshold:
        POL = np.sign(deltaL)
        # 对应的 event accumulate
        spikes = deltaL / threshold  #

        spike_pos = np.where(spike_nums != 0)[0]  # find how many pixels trigger events and where

        for spike_id in range(spike_pos.size):
            pixel_id = spike_pos[spike_id]  # find the pixel_id for spike_id-th event
            spike_num = spike_nums[pixel_id]  # spiked times for this pixels with sign

            spike_num_ = np.abs(spike_num)

            # for num in range(np.abs(spike_num)):
            #     spike_time[pixel_id] += (current_time[pixel_id] - last_time[pixel_id]) / spikes[pixel_id] * POL[pixel_id]
            #     eve_temp = np.array([pixel_id % n_pix_row, pixel_id // n_pix_row, spike_time[pixel_id], POL[pixel_id]]).reshape(1,-1)
            #     eve_gene = np.concatenate([eve_gene, eve_temp], 0)

            spike_time_temp = (spike_time[pixel_id] + (1 + np.arange(spike_num_)) * (
                        current_time[pixel_id] - last_time[pixel_id]) / spikes[pixel_id] * POL[pixel_id]).reshape(-1, 1)
            xs_temp = np.array([[pixel_id % n_pix_row]]).repeat(spike_num_, axis=0)
            ys_temp = np.array([[pixel_id // n_pix_row]]).repeat(spike_num_, axis=0)
            pol_temp = np.array([[POL[pixel_id]]]).repeat(spike_num_, axis=0)

            eve_temp = np.concatenate([xs_temp, ys_temp, spike_time_temp, pol_temp], 1)
            eve_gene = np.concatenate([eve_gene, eve_temp], 0)

        last_time[pixel_id] = spike_time_temp[-1, 0]
        # save the image after generate the event, and this image will be the next base image, cuz the asyn- trigger
        img_base = (img_pre + log_eps) * np.exp(spike_nums * threshold) - log_eps
        # img_base = img_next

        print(i)

        eve_gene_sorted = eve_gene[eve_gene[:, 2].argsort(), :]

        event_file_name = os.path.join(eventdir, '{:03d}.txt'.format(i))

        # with open(event_file_name, "ab") as f:    # 
        with open(event_file_name, "wb") as f:
            np.savetxt(f, eve_gene_sorted, delimiter=" ", fmt='%.8f')

        event_file_name = os.path.join(eventdir, '{:03d}.npy'.format(i))
        np.save(event_file_name, eve_gene_sorted)
    """"""

    # events = np.zeros([0, 4])

    # for i in range(length):
    #     event_file_name = os.path.join(eventdir, '{:03d}.npy'.format(i))
    #     Eve = np.load(event_file_name)
    #     events = np.concatenate([events, Eve], 0)
    # np.save(os.path.join(eventdir, 'gray_events_data.npy'), events)

    for i in range(NUM_I_RGB):
        events = np.zeros([0, 4])
        for j in range(INTERVAL):
            frame_id = i * INTERVAL + j
            event_file_name = os.path.join(eventdir, '{:03d}.npy'.format(frame_id))
            Eve = np.load(event_file_name)
            events = np.concatenate([events, Eve], 0)
        np.save(os.path.join(eventdir, 'gray_events_data_{:03d}.npy'.format(i)), events)


if __name__ == '__main__':

    # for i in range(5):
    #     print(i)

    # exit()
    # event_data = np.load('./data/Event-NeRF/Synthetic-data/Living-Room/events_data.npy')
    # event_data = np.load('./data/Event-NeRF/Synthetic-data/Living-Room/events_data.npy')

    # events = np.zeros([0, 4])
    # event_files_name = [os.path.join(eventdir, f) for f in sorted(os.listdir(eventdir)) if f.endswith('.npy')] 
    # for npy_name in event_files_name:
    #     Eve = np.load(npy_name)
    #     events = np.concatenate([events, Eve], 0)

    # events_sorted = events[events[:,2].argsort(), :]
    # np.save(os.path.join(eventdir, 'gray_events_data.npy'), events_sorted)

    # exit()

    img_id = 0
    for i in range(NUM_I_RGB + 1):
        RGB_dir = imgfiles[img_id]
        I_Gray = cv2.imread(RGB_dir, cv2.IMREAD_GRAYSCALE)
        dir = os.path.join(savedir_Gray, '{:03d}.png'.format(i))
        imageio.imwrite(dir, I_Gray)
        img_id += INTERVAL

    img_id = 0
    for i in range(NUM_I_RGB + 1):
        RGB_dir = imgfiles[img_id]
        I_RGB = imageio.v3.imread(RGB_dir)
        dir = os.path.join(savedir_RGB, '{:03d}.png'.format(i))
        imageio.imwrite(dir, I_RGB)
        img_id += INTERVAL

    img0 = cv2.imread(imgfiles[0], cv2.IMREAD_GRAYSCALE)

    [H, W] = img0.shape
    # Color_Event_Simulate(imgs)

    # imgs_gray = 0.299 * imgs[..., 0] + 0.587 * imgs[..., 1] + 0.114 * imgs[..., 2]
    # imgs_gray_flat = imgs_gray.reshape(NUM_Img, H*W)
    Gray_Event_Simulate(imgfiles, W)

    # Color_Event_Simulate(imgs_gray.reshape(img_num, H, W, 1).repeat(3, axis=3))

    print('end!')
