import os.path

import cv2
import numpy as np

if __name__ == '__main__':
    imgdir = os.path.expanduser(r"D:\blender\scripts\BlenderCV\gttanabata_test1\raw")
    outdir = os.path.expanduser(r"D:\blender\scripts\BlenderCV\gttanabata_test1\raw_half_resolution")
    os.makedirs(outdir, exist_ok=True)
    rate = 2

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgfiles = [f for f in imgfiles if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]

    img = cv2.imread(imgfiles[0])

    hw = img.shape[:2]

    h2 = hw[0] // rate
    w2 = hw[1] // rate

    for i, f in enumerate(imgfiles):
        name = os.path.splitext(os.path.basename(f))[0]
        if os.path.exists(os.path.join(outdir, f"{name}.png")):
            continue
        img = cv2.imread(f)
        img_out = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(outdir, f"{name}.png"), img_out)

    print("Finished")
