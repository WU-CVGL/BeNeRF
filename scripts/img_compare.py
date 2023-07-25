import os

import numpy as np
from imageio.v3 import imread, imwrite

IMG0 = os.path.expanduser("~/Desktop/Test/image_mid.jpg")
IMG1 = os.path.expanduser("C:\\Users\\User\\Desktop\\Test\\gt\\018.png")
IMG_DIFF = os.path.expanduser("C:\\Users\\User\\Desktop\\Test\\out\\out.png")
IMG = os.path.expanduser("C:\\Users\\User\\Desktop\\Test\\out\\out_img.png")

if __name__ == '__main__':
    # im0 = imread(IMG0)
    # im1 = imread(IMG1)
    #
    # diff = np.absolute(im0 - im1)
    #
    # imwrite(IMG, diff)
    #
    imgs = os.path.expanduser("C:\\Users\\User\\Desktop\\Test")

    imgs = [os.path.join(imgs, f) for f in sorted(os.listdir(imgs)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgs = [imread(i, mode="L").astype(np.float32) for i in imgs]

    img = 0
    for i in imgs:
        img += i

    img /= len(imgs)

    img = img.astype(np.uint8)

    im1 = imread(IMG1, mode="L")

    diff = np.absolute(img - im1)
    imwrite(IMG_DIFF, diff, mode="L")
    imwrite(IMG, img, mode="L")