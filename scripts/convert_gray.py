import os.path

import cv2

if __name__ == '__main__':
    indir = os.path.expanduser("D:\\EXP_ORIGINAL\\REAL\\blur_demo\\images_undistorted_blur")
    outdir = os.path.expanduser("D:\\EXP_ORIGINAL\\REAL\\blur_demo\\images_undistorted_blur_gray")

    imgfiles = [os.path.join(indir, f) for f in sorted(os.listdir(indir))]
    imgfiles = [f for f in imgfiles if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]

    for i, imgfile in enumerate(imgfiles):
        gray = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(outdir, f"{i:6d}.png"), gray)
