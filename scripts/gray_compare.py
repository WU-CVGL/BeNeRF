import os

import numpy as np
from imageio.v3 import imread, imwrite
from utils.imgutils import to8bit
from utils.mathutils import safelog

if __name__ == '__main__':
    idx = 18
    events = np.load("../data/Living-Room-1000Hz/events/events_data.npy")
    ts = np.loadtxt("../data/Living-Room-1000Hz/poses_ts.txt")

    img_s = imread("C:\\Users\\User\\Desktop\\TestGray\\1.png", mode="L") / 255.
    img_e = imread("C:\\Users\\User\\Desktop\\TestGray\\4.png", mode="L") / 255.
    h = img_s.shape[0]
    w = img_s.shape[1]

    events = [event for event in events if ts[idx] <= event[2] <= ts[idx + 1]]

    events = np.stack(events)
    print("acc events")


    def accumulate_events(events):
        yx = np.zeros((h, w), dtype=np.float32)
        for event in events:
            x = int(event[0])
            y = int(event[1])
            p = event[3]
            yx[y][x] += p * .1

        return yx


    acc = accumulate_events(events)

    e_s = np.log(img_e + 1e-4) - np.log(img_s + 1e-4)

    out_acc = os.path.expanduser("C:\\Users\\User\\Desktop\\TestGray\\out\\acc.png")
    out_e_s = os.path.expanduser("C:\\Users\\User\\Desktop\\TestGray\\out\\es.png")
    out_diff = os.path.expanduser("C:\\Users\\User\\Desktop\\TestGray\\out\\diff.png")
    imwrite(out_acc, to8bit(acc), mode="L")
    imwrite(out_e_s, to8bit(e_s), mode="L")
    imwrite(out_diff, to8bit(np.abs(img_s * np.exp(acc) - img_e)), mode="L")
