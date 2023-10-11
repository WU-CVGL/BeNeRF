import cv2
import os


def find_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def crop_img(files, x_start, x_end, y_start, y_end):
    for m in range(len(files)):
        path_img = files[m]
        img = cv2.imread(path_img)
        for n in range(len(x_start)):
            crop = img[y_start[n]:y_end[n], x_start[n]:x_end[n]]
            path_, file_ = os.path.splitext(path_img)
            save_path = path_ + '_{}'.format(n) + file_
            cv2.imwrite(save_path, crop)
            if n == 0:
                # B, G, R = 202, 177, 103
                B, G, R = 7, 7, 225
            elif n == 1:
                B, G, R = 244, 6, 135
            elif n == 2:
                # B, G, R = 10, 158, 243
                B, G, R = 0, 255, 0
            elif n == 3:
                B, G, R = 202, 177, 103
            else:
                B, G, R = 10, 158, 243
            cv2.rectangle(img, (x_start[n], y_start[n]), (x_end[n], y_end[n]), (B, G, R), 5)
            if n == len(x_start)-1:
                save_path_img = path_ + '_process' + file_
                cv2.imwrite(save_path_img, img)


if __name__ == '__main__':
    # sup decoration
    # x_start = [411, 237, 89]
    # y_start = [175, 250, 7]
    # x_end = [599, 379, 182]
    # y_end = [300, 344, 193]

    # sup stair, coordinates
    x_start = [0, 166]
    y_start = [94, 62]
    x_end = [77, 471]
    y_end = [190, 266]

    path = './test'  # image folder path

    files = find_files(path)

    crop_img(files, x_start, x_end, y_start, y_end)