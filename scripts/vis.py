import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

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


def vis_3d(events, height, width):
    n = int(3e3)
    ts = np.random.rand(n) * 10
    x = events[:0]
    y = events[:1]
    p = events[:3]
    fig = plt.figure(dpi=800)
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b' if p_ == 1 else 'r' for p_ in p]

    ax.scatter3D(ts, x, y, c=colors, s=1)
    ax.view_init(elev=10, azim=270)  # 更改视角的仰角和方位角
    ax.set_box_aspect([4, 1, 1])
    # ax.set_axis_off()
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # 设置x轴线为透明
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # 设置z轴线为透明
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # 设置z轴线为透明

    plt.show()


if __name__ == '__main__':
    events = np.load("/Users/pianwan/Downloads/blur_1019/carpet/2023-10-17-23-04-44/events/events.npy")
    save_dir = "/Users/pianwan/Downloads/blur_1019/carpet/2023-10-17-23-04-44/output"  # image save path
    ts = 1.697555087649756908e+09
    te = 1.697555087726284504e+09
    duration = (te - ts) / 18
    te = ts + duration

    # ts = 1.697554819352672577e+09
    # te = 1.697554819433324575e+09
    events = np.array([event for event in events if ts <= event[2] <= te])
    # events =  # Nx4 ndarray
    i = 1

    e_img = dvsarray_to_image(events, 640, 480)
    cv.imwrite(f"{save_dir}/{i:0>10d}.png", cv.cvtColor(e_img, cv.COLOR_RGB2BGR))
