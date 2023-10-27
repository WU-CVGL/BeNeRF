import os

import cv2
import numpy as np
import torch

from utils import eventutils
from utils.imgutils import rgb2gray
from utils.mathutils import safelog

if __name__ == '__main__':
    img0 = cv2.imread(r"/Users/pianwan/Desktop/temp/img_000.png", cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(r"/Users/pianwan/Desktop/temp/img_001.png", cv2.IMREAD_GRAYSCALE)

    width = 640
    height = 480
    ts = 1.697555087649756908e+09
    te = 1.697555087726284504e+09
    duration = (te - ts) / 18
    events = np.load("/Users/pianwan/Downloads/blur_1019/carpet/2023-10-17-23-04-44/events/events.npy")
    events = np.array([event for event in events if ts <= event[2] <= ts + duration])

    map = np.zeros((height, width))
    eventutils.accumulate_events_no_numba(map, events[:, 0].astype(int), events[:, 1].astype(int), events[:, 3])

    save_dir = "/Users/pianwan/Downloads/blur_1019/carpet/2023-10-17-23-04-44/output"

    img0 = torch.tensor(img0)
    img1 = torch.tensor(img1)

    diff = safelog(img1) - safelog(img0)
    cv2.imwrite(os.path.join(save_dir, "brightness.png"), np.clip(np.absolute(diff.numpy() * 255), 0, 255))
    cv2.imwrite(os.path.join(save_dir, "accumulate.png"), np.clip(np.absolute(map * 0.1 * 255), 0, 255))
    cv2.imwrite(os.path.join(save_dir, "difference.png"), np.clip(np.absolute(diff.numpy() * 255 - map * 0.1 * 255), 0, 255))
