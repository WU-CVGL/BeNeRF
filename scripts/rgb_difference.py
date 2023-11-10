import os

import cv2
import numpy as np

if __name__ == '__main__':
    in_dir = os.path.expanduser(r"C:\Users\User\PycharmProjects\EventBADNeRF\logs\cup\35\images_test\img_test_080000")
    out_dir = os.path.expanduser(r"C:\Users\User\Desktop\out\diff")
    gt_file = os.path.expanduser(r"C:\Users\User\PycharmProjects\EventBADNeRF\data\cup\images\000035.png")
    files = [f for f in os.listdir(in_dir) if f.lower().endswith((".jpg", "png"))]
    imgs = [cv2.imread(os.path.join(in_dir, f), cv2.IMREAD_GRAYSCALE) for f in files]

    syn = 0
    for i in imgs:
        syn += i.astype(int)

    syn = syn / len(imgs)
    cv2.imwrite(os.path.join(out_dir, "synthesized.png"), np.clip(syn, 0, 255).astype(np.uint8))

    gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
    diff = np.absolute(syn - gt)
    cv2.imwrite(os.path.join(out_dir, "diff.png"), np.clip(diff * 10, 0, 255).astype(np.uint8))
