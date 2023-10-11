import os.path
import cv2
import numpy as np

if __name__ == '__main__':
    input = os.path.expanduser("~/Downloads/pic.png")
    output = os.path.expanduser("~/Downloads/pic_out.png")
    img_0 = cv2.imread(input, cv2.IMREAD_GRAYSCALE)

    img_0 = np.log(img_0 + 1.) * 16.
    img_0[img_0 == 0] = 255
    cv2.imwrite(output, img_0)
