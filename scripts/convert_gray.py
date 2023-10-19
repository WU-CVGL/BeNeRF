import os.path

import cv2

if __name__ == '__main__':
    indir = os.path.expanduser(r"I:\wp-event\real\board\images")
    outdir = os.path.expanduser(r"I:\wp-event\real\board\images_gray")

    imgfiles = [os.path.join(indir, f) for f in sorted(os.listdir(indir))]
    imgfiles = [f for f in imgfiles if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]

    os.makedirs(outdir, exist_ok=True)

    for i, imgfile in enumerate(imgfiles):
        name = os.path.splitext(os.path.basename(imgfile))[0]
        gray = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(outdir, f"{name}.png"), gray)
