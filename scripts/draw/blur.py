import cv2
import os
import numpy as np


def find_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def blur_img(files, avg_num):
    for m in range(len(files)):
        path_img = files[m]
        img = cv2.imread(path_img)
        img = img.astype(np.float32)
        width = img.shape[1]
        imgs = 0
        for n in range(avg_num):
            imgs += img[:, n: width - avg_num + n]
            if n == avg_num - 1:
                img_blur = imgs / avg_num
                path_, file_ = os.path.splitext(path_img)
                save_path = path_ + '_blur' + file_
                cv2.imwrite(save_path, img_blur)


if __name__ == '__main__':
    path = './test'  # image folder path

    files = find_files(path)

    blur_img(files, avg_num=10)
