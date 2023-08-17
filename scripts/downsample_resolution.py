import os.path

import cv2
import numpy as np

if __name__ == '__main__':
    imgdir = os.path.expanduser("D:\\EXP_ORIGINAL\\LivingRoom\\camera\\temp")
    outdir = os.path.expanduser("D:\\EXP_ORIGINAL\\LivingRoom\\camera\\half_reso")
    rate = 2

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgfiles = [f for f in imgfiles if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]

    imgs = [cv2.imread(f) for f in imgfiles]

    hw = imgs[0].shape[:2]

    h2 = hw[0] // rate
    w2 = hw[1] // rate

    for i, img in enumerate(imgs):
        img_out = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(outdir, f"{i:06d}.png"), img_out)

    print("Finished")
