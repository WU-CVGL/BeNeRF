import collections
import os
from typing import List

import cv2
from imageio.v3 import imread, imwrite

from spline import *

BASE_DIR: str = os.path.expanduser("C:\\Users\\User\\PycharmProjects\\EventBADNeRF\\data\\office_zigzag")
IMAGE_DIR: str = os.path.expanduser("C:\\Users\\User\\PycharmProjects\\EventBADNeRF\\data\\office_zigzag\\images")
INTRINSICS: str = os.path.join(BASE_DIR, "camera", "intrinsics.txt")
EXTRINSICS: str = os.path.join(BASE_DIR, "camera", "extrinsics.txt")
OUTPUT_DIR: str = os.path.join(BASE_DIR,
                               "C:\\Users\\User\\PycharmProjects\\EventBADNeRF\\data\\office_zigzag\\images_blur")
TEST_DIR: str = os.path.join(BASE_DIR,
                             "C:\\Users\\User\\PycharmProjects\\EventBADNeRF\\data\\office_zigzag\\images_test")
GROUNDTRUTH_POSE: str = os.path.join(BASE_DIR, "groundtruth.txt")
BLUR_NUM: int = 51
DATASET_NAME = 'Living_Room_1000Hz'

type = "esim"


def Spline_GT(start_pose, end_pose, poses_number, H):
    # start_pose & end_pose are se3

    # spline t
    pose_list = []
    # pose_time = poses_number.reshape([-1, 1])/(H-1)    # 选择的 pose 对应行数 / 总行数 = 此时对应的 T

    pose_time = poses_number / (H - 1)

    # parallel

    pos_0 = torch.where(pose_time == 0)
    pose_time[pos_0] = pose_time[pos_0] + 0.000001
    pos_1 = torch.where(pose_time == 1)
    pose_time[pos_1] = pose_time[pos_1] - 0.000001

    # parallel
    # pose_time = pose_time.reshape([-1, 1])   # [6] --> [6,1]
    q_start = start_pose[..., 4:]
    t_start = start_pose[..., 1:4]
    q_end = end_pose[..., 4:]
    t_end = end_pose[..., 1:4]

    # sample t_vector
    t_t = (1 - pose_time)[..., None] * t_start + pose_time[
        ..., None] * t_end  # [35, 6, 3] (35 imgs * 6 poses_per_img * 3 dims)
    # print(t_t[0], t_t[1], t_t[2])

    # sample rotation_vector
    q_tau_0 = q_to_Q_parallel(q_to_q_conj_parallel(q_start)) @ q_end[..., None]  # [35, 4, 1]  # equation 50 shape:[4]
    r = pose_time[..., None] * log_q2r_parallel(
        q_tau_0.squeeze(-1))  # [35, 6, 3]   # [6,1] * [35, 1, 3] = [35, 6, 3]  # equation 51 shape:[3]
    q_t_0 = exp_r2q_parallel(r)  # equation 52 shape:[4]    # [35, 6, 4]
    q_t = q_to_Q_parallel(q_start) @ q_t_0[..., None]  # [35, 6, 4, 1]  # equation 53 shape:[4]

    # convert q&t to RT
    R = q_to_R_parallel(q_t.squeeze(dim=-1))  # [3,3]    # [35, 6, 3, 3]
    t = t_t.unsqueeze(dim=-1)  # [35, 6, 3, 1]
    pose_spline = torch.cat([R, t], -1)  # [3, 4]

    # poses = pose_spline.reshape([-1, 3, 4])  # [35, 6, 3, 4]

    return pose_spline


Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
OriginalExtrinsic = collections.namedtuple(
    'Extrinsic', ['id', 'qvec', 'tvec'])
OriginalGroundTruth = collections.namedtuple(
    'Extrinsic', ['id', 'qvec', 'tvec'])
Depth = collections.namedtuple(
    'Depth', ['id', 'min', 'max'])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


class Extrinsic(OriginalExtrinsic):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_extrinsics(path):
    extrinsic = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))  # wxyz
                tvec = np.array(tuple(map(float, elems[5:8])))
                extrinsic[camera_id] = Extrinsic(
                    id=camera_id, qvec=qvec, tvec=tvec
                )
    return extrinsic


def read_intrinsics(path):
    cameras = {}
    with open(path, 'r') as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def poses_2_mat(groundtruth_data, m_c2b, near, far, hwf, bottom):
    c2w_mats = []
    for k in range(groundtruth_data.shape[0]):
        groundtruth = groundtruth_data[k]
        R_b2w, T_b2w = qvec2rotmat(groundtruth[[7, 4, 5, 6]]), groundtruth[1:4].reshape(3, 1)
        m_b2w = np.concatenate([np.concatenate([R_b2w, T_b2w], 1), bottom], 0)
        # print(m_b2w)
        m_c2w = m_b2w @ m_c2b

        c2w_mats.append(m_c2w)
    c2w_mats = np.stack(c2w_mats, 0)

    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1)
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t] ???
    # poses [3, 5, N]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]],
                           1)

    poses_bounds = []
    for i in range(0, poses.shape[2]):

        # near, far = depth_data[i][0], depth_data[i][1]
        if DATASET_NAME == 'blue_room':
            near, far = 0.5, 121.51878159270797

        poses_bounds.append(np.concatenate([poses[..., i].ravel(), np.array([near, far])], 0))
    poses_bounds = np.array(poses_bounds)

    return poses_bounds


def cal_bounds(basedir):
    data_dir = os.path.join(basedir, 'camera/temp')

    near = np.ones([0, 1])
    far = np.ones([0, 1])
    imgfiles = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))]
    imgfiles = [f for f in imgfiles if any([f.endswith(ex) for ex in ['exr']])]

    H = 480
    num_per_seq = 10
    sky_depth = 500
    seq_num = len(imgfiles) // H

    file_idx = np.arange(seq_num * num_per_seq) * int(H / num_per_seq)
    imgfiles = [imgfiles[i] for i in file_idx]

    for filename in imgfiles:
        # image = imageio.imread(filename, 'exr')
        # image = imageio.imread(filename)
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        near = np.concatenate([near, np.array([[image[:, :, 0].min()]])])
        img_no_sky = image[np.where(image[:, :, 0] < sky_depth)[0], np.where(image[:, :, 0] < sky_depth)[1]]
        far = np.concatenate([far, np.array([[img_no_sky[:, 0].max()]])])

    print('Near: ', near.min(), '\n')
    print('Far: ', far.max(), '\n')

    return near.min(), far.max()


def check_dir(dirs: List[str]) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def main():
    print("-> Now loading files")
    image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)]
    image_files = [f for f in image_files if f.lower().endswith(("jpg", "jpeg", "png"))]
    image_files = [f for f in sorted(image_files)]

    if False:
        gt_poses = np.loadtxt(GROUNDTRUTH_POSE)
        print("-> Generating poses")
        intrinsics = read_intrinsics(INTRINSICS)
        extrinsics = read_extrinsics(EXTRINSICS)
        h, w, f = intrinsics[0].height, intrinsics[0].width, intrinsics[0].params[0],
        hwf = np.array([h, w, f]).reshape(3, 1)

        start_id = np.arange(gt_poses.shape[0] // BLUR_NUM) * BLUR_NUM
        end_id = np.arange(gt_poses.shape[0] // BLUR_NUM) * BLUR_NUM + BLUR_NUM - 1
        mid_id = np.arange(gt_poses.shape[0] // BLUR_NUM) * BLUR_NUM + BLUR_NUM // 2
        bound_id = np.arange(gt_poses.shape[0] // BLUR_NUM + 1) * BLUR_NUM

        R_c2b, T_c2b = extrinsics[0].qvec2rotmat(), extrinsics[0].tvec.reshape(3, 1)
        bottom = np.array([0, 0, 0, 1]).reshape([1, 4])
        m_c2b = np.concatenate([np.concatenate([R_c2b, T_c2b], 1), bottom], 0)

        near, far = cal_bounds(BASE_DIR)

        poses_GT_bound = poses_2_mat(gt_poses, m_c2b, near, far, hwf, bottom)

        pose_start = poses_GT_bound[start_id]
        pose_end = poses_GT_bound[end_id]
        pose_mid = poses_GT_bound[mid_id]
        pose_bound = poses_GT_bound[bound_id]

        pose_ts = gt_poses[bound_id, 0]

        np.save(os.path.join(BASE_DIR, "pose_bound_start.npy"), pose_start)
        np.save(os.path.join(BASE_DIR, "pose_bound_end.npy"), pose_end)
        np.save(os.path.join(BASE_DIR, "pose_bound_mid.npy"), pose_mid)
        np.save(os.path.join(BASE_DIR, "pose_bounds.npy"), pose_bound)
        np.save(os.path.join(BASE_DIR, "pose_ts.npy"), pose_ts)

    if type == "davis":
        print("Loading davis timestamps")
        img_ts = np.loadtxt(os.path.join(BASE_DIR, "images.txt"), usecols=0)
        ts = []
        for idx in range(len(image_files)):
            if (idx + 1) % BLUR_NUM == 0 or idx == 0:
                ts.append(img_ts[idx])

        ts = np.stack(ts)
        np.savetxt(os.path.join(BASE_DIR, "poses_ts.txt"), ts)
        print("Saved timestamps")

    elif type == "esim":
        print("Loading esim timestamps")
        
        print("Saved timestamps")
    else:
        print("Undefined dataset")
        return


    print("-> Blurring images")
    print(f"BLUR_NUM: {BLUR_NUM}")
    img = 0
    res = 0
    amount = 0
    idx_blur = []
    for idx in range(len(image_files)):
        img += imread(image_files[idx]).astype(np.float32)
        if (idx + 1) % BLUR_NUM == 0:
            img /= np.array(BLUR_NUM).astype(np.float32)
            img = img.round().astype(np.uint8)
            imwrite(os.path.join(OUTPUT_DIR, "{:0>3d}.jpg".format(amount)), img)
            img = 0
            amount += 1
            idx_blur.append(idx)

        res = (idx + 1) % BLUR_NUM

    amount_sharp = 0
    idx_sharp = []
    for idx in range(len(image_files)):
        if idx % BLUR_NUM == BLUR_NUM // 2 and (idx + BLUR_NUM // 2 < len(image_files)):
            imwrite(os.path.join(TEST_DIR, "{:0>3d}.jpg".format(amount_sharp)), imread(image_files[idx]))
            idx_sharp.append(idx)
            amount_sharp += 1

    print(f"Redundant images files: {res}")
    print(f"Blur images files: {amount} bounds {idx_blur}")
    print(f"Shap images files: {amount_sharp} in {idx_sharp}")


if __name__ == '__main__':
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    check_dir([IMAGE_DIR, OUTPUT_DIR, TEST_DIR])
    main()
