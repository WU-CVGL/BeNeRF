import os

import numpy as np
from imageio.v3 import imread, imwrite

IMG0 = os.path.expanduser("~/Desktop/Test/image_mid.jpg")
IMG1 = os.path.expanduser("~/Desktop/Test/018.jpg")
IMG = os.path.expanduser("~/Desktop/Test/out.jpg")

if __name__ == '__main__':
    im0 = imread(IMG0)
    im1 = imread(IMG1)

    diff = np.absolute(im0 - im1)

    imwrite(IMG, diff)
