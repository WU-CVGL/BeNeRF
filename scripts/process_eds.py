import glob
import os

import h5py

eds_dir = os.path.expanduser("C:\\Users\\User\\PycharmProjects\\EventBADNeRF\\data\\PROCESS\\02_rocket_earth_light")

if __name__ == '__main__':
    h5file = os.path.join(eds_dir, "events.h5")
    events = h5py.File(h5file, "r")
    keys = events.keys()
    for v in events.values():
        p = v['p']
        x = v['x']
        y = v['y']
        t = v['t']
    print()