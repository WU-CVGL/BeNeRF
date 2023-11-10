import os

import cv2
import numpy as np

if __name__ == '__main__':
    in_dir = os.path.expanduser(r"")
    out_dir = os.path.expanduser(r"")
    gt_file = os.path.expanduser(r"")
    files = [f for f in os.listdir(in_dir) if f.lower().endswith((".jpg", "png"))]
    imgs = [cv2.imread(os.path.join(in_dir, f)) for f in files]

    syn = 0
    for i in imgs:
        syn += i

    syn /= len(imgs)
    cv2.imwrite(os.path.join(out_dir, "synthesized.png"), syn)

    gt = cv2.imread(gt_file)
    diff = np.absolute(syn - gt)
    cv2.imwrite(os.path.join(out_dir, "diff.png"), diff)
