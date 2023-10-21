import cv2 as cv
import numpy as np

from utils import eventutils


def dvsarray_to_image(events, height, width):
    # Draw event image in RPG way
    dvs_image = np.ones((width, height, 3), dtype=np.dtype("uint8"))
    dvs_image *= 255
    map = np.zeros((width, height))
    eventutils.accumulate_events_no_numba(map, events[:, 0].astype(int), events[:, 1].astype(int), events[:, 3])
    pos_map = map > 0
    neg_map = map < 0
    dvs_image[pos_map] = [0, 0, 255]
    dvs_image[neg_map] = [255, 0, 0]
    return dvs_image


if __name__ == '__main__':
    events = np.load("/Users/pianwan/Downloads/blur_1019/carpet/2023-10-17-23-04-19/events/events.npy")
    save_dir = "/Users/pianwan/Downloads/blur_1019/carpet/2023-10-17-23-04-19/output"  # image save path
    ts = 1.697555066208295822e+09
    te = 1.697555066284142733e+09
    # ts = 1.697554819352672577e+09
    # te = 1.697554819433324575e+09
    events = np.array([event for event in events if ts <= event[2] <= te])
    # events =  # Nx4 ndarray
    i = 1

    e_img = dvsarray_to_image(events, 640, 480)
    cv.imwrite(f"{save_dir}/{i:0>10d}.png", cv.cvtColor(e_img, cv.COLOR_RGB2BGR))
