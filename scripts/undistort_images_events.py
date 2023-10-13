import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import tqdm


def process_timestamp(dir, imgdir):
    csv_file = os.path.join(imgdir, "ts_frame.csv")
    ts_start_file = os.path.join(dir, "poses_start_ts.txt")
    ts_end_file = os.path.join(dir, "poses_end_ts.txt")
    timestamp_csv = pd.read_csv(csv_file, sep=" ", header=None)
    ts = timestamp_csv[1].values
    ts_start = np.insert(ts, 0, ts[0])[:-1]
    ts_end = np.insert(ts, len(ts), ts[-1])[1:]
    np.savetxt(ts_start_file, ts_start)
    np.savetxt(ts_end_file, ts_end)


def main():
    parser = argparse.ArgumentParser(description="Resizes images in dir")
    parser.add_argument(
        "--indir", help="Input image directory.", default=r"D:\real\temp\outdoor_1010\extracted\2023-10-10-16-14-32"
    )
    parser.add_argument(
        "--h_rgb", type=int, default=1080
    )
    parser.add_argument(
        "--w_rgb", type=int, default=1440
    )
    parser.add_argument(
        "--h_ev", type=int, default=480
    )
    parser.add_argument(
        "--w_ev", type=int, default=640
    )
    args = parser.parse_args()

    datastr = "blur"
    print(f"Processing {args.indir}")
    W, H = args.w_rgb, args.h_rgb
    W_ev, H_ev = args.w_ev, args.h_ev

    evdir = os.path.join(args.indir, "events")
    imgdir = os.path.join(args.indir, "rgb")
    imgdirout = os.path.join(args.indir, f"images_undistorted_{datastr}")
    print("Process timestamps...")
    process_timestamp(args.indir, imgdir)

    print("Undistorting...")
    os.makedirs(imgdirout, exist_ok=True)
    img_list = sorted(os.listdir(os.path.join(args.indir, imgdir)))
    img_list = [os.path.join(args.indir, imgdir, im) for im in img_list if im.endswith(".png")]

    event_list = sorted(os.listdir(os.path.join(args.indir, evdir)))
    event_list = [np.loadtxt(os.path.join(args.indir, evdir, f)) for f in event_list if f.endswith("txt")]
    event_list = np.vstack(event_list)
    events = np.moveaxis(event_list, 0, -1)

    # calib data
    K_rgb = np.zeros((3, 3))
    if datastr == "blur":
        K_rgb[0, 0] = 1.782310233143170e+03
        K_rgb[0, 2] = 6.830735988620997e+02
        K_rgb[1, 1] = 1.782887779096247e+03
        K_rgb[1, 2] = 5.964069032198300e+02
        K_rgb[2, 2] = 1
        dist_coeffs_rgb = np.asarray(
            [-0.200569353156535, 0.129469865111905, 0., 0.])

    K_new_rgb, roi = cv2.getOptimalNewCameraMatrix(K_rgb, dist_coeffs_rgb, (W, H), alpha=0, newImgSize=(
        W, H))  # alpha = 0 => all pixels in undistorted image are valid
    x, y, w, h = roi
    assert x == 0 and y == 0 and w + 1 == W and h + 1 == H
    intr_undist = []
    intr_undist.append({"fx": K_new_rgb[0, 0], "fy": K_new_rgb[1, 1], "cx": K_new_rgb[0, 2], "cy": K_new_rgb[1, 2]})

    K_evs = np.zeros((3, 3))
    if datastr == "blur":
        K_evs[0, 0] = 6.857668012086921e+02
        K_evs[0, 2] = 3.314548914718735e+02
        K_evs[1, 1] = 6.871234535722497e+02
        K_evs[1, 2] = 2.357054816902933e+02
        K_evs[2, 2] = 1
        dist_coeffs_evs = np.asarray(
            [-0.263867253487239, 0.452167567807637, 0., 0.])

    K_new_evs, roi = cv2.getOptimalNewCameraMatrix(K_evs, dist_coeffs_evs, (W_ev, H_ev), alpha=0,
                                                   newImgSize=(W_ev, H_ev))
    x, y, w, h = roi
    assert x == 0 and y == 0 and w + 1 == W_ev and h + 1 == H_ev
    intr_undist.append({"fx": K_new_evs[0, 0], "fy": K_new_evs[1, 1], "cx": K_new_evs[0, 2], "cy": K_new_evs[1, 2]})

    with open(os.path.join(args.indir, f"calib_undist_{datastr}.json"), 'w') as f:
        calibdata = {}
        calibdata["intrinsics_undistorted"] = intr_undist
        json.dump(calibdata, f)

    print("Undistorting images")
    pbar = tqdm.tqdm(total=len(img_list))
    for f in img_list:
        image = cv2.imread(f)
        img = cv2.undistort(image, K_rgb, dist_coeffs_rgb, newCameraMatrix=K_new_rgb)
        cv2.imwrite(os.path.join(imgdirout, os.path.split(f)[1]), img)
        pbar.update(1)

    # Computing joint recitificaiton
    K_joint = K_evs  # (K_evs + K_rgb) / 2.
    # img_mapx, img_mapy = cv2.initUndistortRectifyMap(K_rgb, dist_coeffs_rgb, newR, K_joint, (W, H), cv2.CV_32FC1)
    ev_mapx, ev_mapy = cv2.initUndistortRectifyMap(K_evs, dist_coeffs_evs, np.eye(3), K_joint, (W, H), cv2.CV_32FC1)

    # tss_img_us = np.loadtxt(os.path.join(args.indir, "images_timestamps_us.txt"))
    # print(f"Visualizing undistorted {len(tss_img_us)} event slices around images (with aligned optical axis)")

    x = events[1].astype(int)
    y = events[2].astype(int)
    x_rect = ev_mapx[y, x]
    y_rect = ev_mapy[y, x]
    events = np.stack((x_rect, y_rect, events[0], events[3]))
    events = np.moveaxis(events, 0, 1)
    idx = np.where(
        (events[:, 0] <= args.w_ev) & (events[:, 0] >= 0) & (events[:, 1] <= args.h_ev) & (events[:, 1] >= 0))
    events = events[idx]

    np.save(os.path.join(args.indir, "events", "events.npy"), events)


if __name__ == "__main__":
    process_timestamp()
