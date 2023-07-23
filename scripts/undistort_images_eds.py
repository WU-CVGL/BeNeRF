import argparse
import json
import math
import os

import cv2
import h5py
import numpy as np
import tqdm


def compute_ms_to_idx(tss_ns, ms_start=0):
    """
    evs_ns: (N, 4)
    idx_start: Integer
    ms_start: Integer
    """

    ms_to_ns = 1000000
    # tss_sorted, _ = torch.sort(tss_ns)
    # assert torch.abs(tss_sorted != tss_ns).sum() < 500

    ms_end = int(math.floor(tss_ns.max()) / ms_to_ns)
    assert ms_end >= ms_start
    ms_window = np.arange(ms_start, ms_end + 1, 1).astype(np.uint64)
    ms_to_idx = np.searchsorted(tss_ns, ms_window * ms_to_ns, side="left", sorter=np.argsort(tss_ns))

    assert np.all(np.asarray([(tss_ns[ms_to_idx[ms]] >= ms * ms_to_ns) for ms in ms_window]))
    assert np.all(np.asarray([(tss_ns[ms_to_idx[ms] - 1] < ms * ms_to_ns) for ms in ms_window if ms_to_idx[ms] >= 1]))

    return ms_to_idx


def main():
    parser = argparse.ArgumentParser(description="Resizes images in dir")
    parser.add_argument(
        "--indir", help="Input image directory.", default="DATADIR/00_peanuts_dark/"
    )
    args = parser.parse_args()

    calibstr = "calib0"
    assert calibstr == "calib0" or calibstr == "calib1"
    print(f"Processing {args.indir}")

    evinfile = os.path.join(args.indir, "events.h5")
    assert os.path.isfile(evinfile)

    imgdir = os.path.join(args.indir, "images")
    imgdirout = os.path.join(args.indir, f"images_undistorted_{calibstr}")
    os.makedirs(imgdirout, exist_ok=True)

    img_list = sorted(os.listdir(os.path.join(args.indir, imgdir)))
    img_list = [os.path.join(args.indir, imgdir, im) for im in img_list if im.endswith(".png")]
    H, W, _ = cv2.imread(img_list[0]).shape
    assert W == 640
    assert H == 480

    # 1) Getting offset which is substracted from evs, mocap and images.
    ef_in = h5py.File(os.path.join(args.indir, evinfile), "r+")
    tss_evs_us = ef_in["t"][:]
    gt_us = np.loadtxt(os.path.join(args.indir, "stamped_groundtruth.txt"))
    tss_gt_us = gt_us[:, 0] * 1e6
    tss_imgs_us = np.loadtxt(os.path.join(args.indir, "images_timestamps.txt"), skiprows=0)

    if not os.path.isfile(os.path.join(args.indir, "t_offset_us.txt")):
        offset_us = np.minimum(tss_evs_us.min(), np.minimum(tss_gt_us.min(), tss_imgs_us.min())).astype(np.int64)
        print(
            f"Minimum/offset_us is {offset_us}. tss_evs_us.min() = {tss_evs_us.min() - offset_us},  tss_gt_us.min() = {tss_gt_us.min() - offset_us}, tss_imgs_us.min() = {tss_imgs_us.min() - offset_us}")
        assert offset_us != 0
        assert offset_us > 0

        tss_gt_us -= offset_us
        gt_us[:, 0] = tss_gt_us
        np.savetxt(os.path.join(args.indir, "stamped_groundtruth_us.txt"), gt_us,
                   header="#timestamp[us] px py pz qx qy qz qw")

        tss_imgs_us -= offset_us
        np.savetxt(os.path.join(args.indir, "images_timestamps_us.txt"), tss_imgs_us, fmt="%d")

        ef_in["t"][:] -= offset_us
        tss_evs_us -= offset_us
        np.savetxt(os.path.join(args.indir, "t_offset_us.txt"), np.array([offset_us]))
    else:
        assert ef_in["t"][0] < 5000

    # calib data
    K_rgb = np.zeros((3, 3))
    if calibstr == "calib1":
        K_rgb[0, 0] = 758.1291471478728  ##### calib1
        K_rgb[0, 2] = 289.0985666049996
        K_rgb[1, 1] = 759.5125594392973
        K_rgb[1, 2] = 228.23374237672056
        K_rgb[2, 2] = 1
        dist_coeffs_rgb = np.asarray(
            [-0.36599825863847607, 0.15566628749131536, 0.003684464282510181, 0.004564651739351755])
    elif calibstr == "calib0":
        K_rgb[0, 0] = 766.536025127154  ##### calib0
        K_rgb[0, 2] = 291.0503512057777
        K_rgb[1, 1] = 767.5749459126396
        K_rgb[1, 2] = 227.4060484950132
        K_rgb[2, 2] = 1
        dist_coeffs_rgb = np.asarray(
            [-0.36965913545735024, 0.17414034009883844, 0.003915245015812422, 0.003666687416655559])

    K_new_rgb, roi = cv2.getOptimalNewCameraMatrix(K_rgb, dist_coeffs_rgb, (W, H), alpha=0, newImgSize=(
        W, H))  # alpha = 0 => all pixels in undistorted image are valid
    x, y, w, h = roi
    assert x == 0 and y == 0 and w + 1 == W and h + 1 == H
    intr_undist = []
    intr_undist.append({"fx": K_new_rgb[0, 0], "fy": K_new_rgb[1, 1], "cx": K_new_rgb[0, 2], "cy": K_new_rgb[1, 2]})

    K_evs = np.zeros((3, 3))
    if calibstr == "calib1":
        K_evs[0, 0] = 548.8989250692618  ##### calib1
        K_evs[0, 2] = 313.5293514832678
        K_evs[1, 1] = 550.0282089284915
        K_evs[1, 2] = 219.6325753720951
        K_evs[2, 2] = 1
        dist_coeffs_evs = np.asarray(
            [-0.08095806072593555, 0.15743578875760092, -0.0035154416164982195, -0.003950567808338846])
    elif calibstr == "calib0":
        K_evs[0, 0] = 560.8520948927032  ##### calib0
        K_evs[0, 2] = 313.00733235019237
        K_evs[1, 1] = 560.6295819972383
        K_evs[1, 2] = 217.32858679842997
        K_evs[2, 2] = 1
        dist_coeffs_evs = np.asarray(
            [-0.09776467241921379, 0.2143738428636279, -0.004710710105172864, -0.004215916089401789])

    K_new_evs, roi = cv2.getOptimalNewCameraMatrix(K_evs, dist_coeffs_evs, (W, H), alpha=0, newImgSize=(W, H))
    x, y, w, h = roi
    assert x == 0 and y == 0 and w + 1 == W and h + 1 == H
    intr_undist.append({"fx": K_new_evs[0, 0], "fy": K_new_evs[1, 1], "cx": K_new_evs[0, 2], "cy": K_new_evs[1, 2]})

    # 1) Saving undistorted intrinsics
    with open(os.path.join(args.indir, f"calib_undist_{calibstr}.json"), 'w') as f:
        calibdata = {}
        calibdata["intrinsics_undistorted"] = intr_undist
        json.dump(calibdata, f)

    # 2) undistorting images
    print("Undistorting images")
    pbar = tqdm.tqdm(total=len(img_list))
    for f in img_list:
        image = cv2.imread(f)
        img = cv2.undistort(image, K_rgb, dist_coeffs_rgb, newCameraMatrix=K_new_rgb)
        cv2.imwrite(os.path.join(imgdirout, os.path.split(f)[1]), img)
        pbar.update(1)
        # for debugging:
        # cv2.imwrite(os.path.join(imgdirout,  os.path.split(f)[1][:-4] + "_undist.jpg"),  image)
    # shutil.copy(os.path.join(imgdir, "timestamps.txt"), os.path.join(imgdirout, "timestamps.txt"))
    # sys.exit()

    # 3) undistorting events => visualize
    coords = np.stack(np.meshgrid(np.arange(W), np.arange(H))).reshape((2, -1)).astype("float32")
    term_criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001)
    points = cv2.undistortPointsIter(coords, K_evs, dist_coeffs_evs, np.eye(3), K_new_evs, criteria=term_criteria)
    rectify_map = points.reshape((H, W, 2))

    # 4) Create rectify map for events
    h5outfile = os.path.join(args.indir, f"rectify_map_{calibstr}.h5")
    ef_out = h5py.File(h5outfile, 'w')
    ef_out.clear()
    ef_out.create_dataset('rectify_map', shape=(H, W, 2), dtype="<f4")
    ef_out["rectify_map"][:] = rectify_map
    ef_out.close()

    # 5) Computing ms_to_idx
    tss_evs_ns = tss_evs_us * 1000
    if "ms_to_idx" not in ef_in.keys():
        print(f"Start computing ms_to_idx, with {len(tss_evs_ns)} tss_evs_ns, {tss_evs_ns}, ms_start={ef_in['t'][0]}")
        ms_to_idx = compute_ms_to_idx(tss_evs_ns)
        print(f"Done computing ms_to_idx")
        ef_in.create_dataset('ms_to_idx', shape=len(ms_to_idx), dtype="<u8")
        ef_in["ms_to_idx"][:] = ms_to_idx


if __name__ == "__main__":
    main()
