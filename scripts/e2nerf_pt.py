import os.path

import cv2
import numpy as np
from dv import AedatFile

if __name__ == '__main__':
    file = r"/Users/pianwan/Downloads/orginal_data&preprocessing/real-world/davis-aedat4/lego.aedat4"
    f = AedatFile(file)

    process_timestamp = True
    if process_timestamp:
        base = r"/Users/pianwan/Downloads/orginal_data&preprocessing/real-world/davis-aedat4/images_o/"
        imgs = f["frames"]
        start_time, end_time = [], []
        for img in imgs:
            ts_st = img.timestamp_start_of_exposure
            ts_ed = img.timestamp_end_of_exposure
            start_time.append(ts_st)
            end_time.append(ts_ed)

        start_time = np.hstack(start_time)
        end_time = np.hstack(end_time)

        np.savetxt(base + "poses_start_ts.txt", start_time)
        np.savetxt(base + "poses_end_ts.txt", end_time)

    process_iamge = False
    if process_iamge:
        imgs = f["frames"]
        base = r"/Users/pianwan/Downloads/orginal_data&preprocessing/real-world/davis-aedat4/images_o/"
        for i, img in enumerate(imgs):
            img_np = img.image
            cv2.imwrite(base + '{:06d}.png'.format(i), img_np)

    process_event = False
    if process_event:
        events = f["events"]
        event_list = []
        for event in events:
            ts = event.timestamp
            p = 1 if event.polarity else -1
            x = event.x
            y = event.y
            event_list.append(np.array([x, y, ts, p]))
        events_np = np.vstack(event_list)
        np.save(
            os.path.expanduser("/Users/pianwan/Downloads/orginal_data&preprocessing/real-world/davis-aedat4/lego.npy"),
            events_np)
