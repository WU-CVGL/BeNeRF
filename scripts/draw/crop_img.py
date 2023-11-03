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
    # livingroom: 18
    # lamp, leaves, painting
    # x_start = [375, 660, 283]
    # y_start = [145, 27, 5]
    # x_end = [420, 737, 353]
    # y_end = [245, 162, 51]

    # tanabata: 5
    # font, ball, lines
    # x_start = [80, 340, 532]
    # y_start = [14, 230, 5]
    # x_end = [190, 378, 590]
    # y_end = [145, 267, 97]

    # outdoorpool: 11
    # x_start = [248, 372]
    # y_start = [172, 310]
    # x_end = [317, 490]
    # y_end = [257, 373]

    # whiteroom: 5
    # x_start = [306, 588]
    # y_start = [297, 81]
    # x_end = [448, 677]
    # y_end = [403, 157]

    # real-carpet-168
    x_start = [851, 379, 179, 1156]
    y_start = [459, 220, 851, 786]
    x_end = [1123, 556, 299, 1377]
    y_end = [625, 319, 982, 1037]

    path = r'C:\Users\User\Downloads\test'  # image folder path

    files = find_files(path)

    crop_img(files, x_start, x_end, y_start, y_end)
