import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from utils import event_utils


def dvsarray_to_image(events, height, width, init_map=None):
    # Draw event image in RPG way
    if init_map is None:
        dvs_image = np.ones((width, height, 3), dtype=np.dtype("uint8"))
        dvs_image *= 255
    else:
        dvs_image = init_map
    map = np.zeros((width, height))
    event_utils.accumulate_events_no_numba(map, events[:, 0].astype(int), events[:, 1].astype(int), events[:, 3])
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
    events = np.load(r"C:\Users\User\PycharmProjects\EventBADNeRF\data\carpet\events\events.npy")
    save_dir = r"C:\Users\User\PycharmProjects\EventBADNeRF\data\carpet\out"  # image save path
    ts = 1.697555087686322927e+09 - 60e-3
    te = ts + 20e-3
    # duration = (te - ts) / 18
    # te = ts + duration

    # ts = 1.697554819352672577e+09
    # te = 1.697554819433324575e+09
    events = np.array([event for event in events if ts <= event[2] <= te])
    # events =  # Nx4 ndarray
    i = 1

    img = cv.imread(r"C:\Users\User\PycharmProjects\EventBADNeRF\data\carpet\images\000065.png")
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
    e_img = dvsarray_to_image(events, 640, 480)
    cv.imwrite(f"{save_dir}/{i:0>10d}.png", cv.cvtColor(e_img, cv.COLOR_RGB2BGR))
    img_all = cv2.addWeighted(e_img, 0.5, img, 0.9, 0.)
    cv.imwrite(f"{save_dir}/{i:0>10d}.png", cv.cvtColor(img_all, cv.COLOR_RGB2BGR))
