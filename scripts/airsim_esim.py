import os
from types import SimpleNamespace

import cv2
import numpy as np

from dataprocess.event_simulator import EventSimulator

CONFIG = SimpleNamespace(
    **{
        "contrast_thresholds": (0.01, 0.01),
        "sigma_contrast_thresholds": (0.0, 0.0),
        "refractory_period_ns": 1e-10,
        "max_events_per_frame": 2000000,
    }
)
imgdir = os.path.expanduser("D:\\dataset\\LivingRoom\\camera\\test")
eventdir = os.path.expanduser("D:\\dataset\\LivingRoom\\camera\\teste")

if __name__ == '__main__':
    np.random.seed(0)
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    H, W = cv2.imread(imgfiles[0]).shape[:2]
    ev_sim = EventSimulator(W, H, config=CONFIG)
    for imgfile in imgfiles:
        img = cv2.imread(imgfile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        img = cv2.add(img, 0.001)

        file_name = os.path.splitext(os.path.basename(imgfile))[0]
        ts_ns = round((float(file_name)) * 1e9)

        event_img, events = ev_sim.image_callback(img, ts_ns)
        print(events)
    # img = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) / 255. for f in imgfiles]
    # img = np.stack(img)
    # event_img, events = ev_sim.image_callback(img, 1)
    # print(events)
