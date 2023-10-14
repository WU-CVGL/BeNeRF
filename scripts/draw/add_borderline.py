from PIL import Image, ImageOps
import os


def find_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith(".DS_Store"):
                file_list.append(os.path.join(root, file))
    return file_list


def add_border(image_path, border_size, border_color):
    for img in image_path:
        with Image.open(img) as im:
            new_im = ImageOps.expand(im, border=border_size, fill=border_color)
            new_im.save(img)


if __name__ == '__main__':
    path = '/Users/pianwan/Desktop/DataProcess/tanabata_2'  # image folder path
    file = find_files(path)
    # image color: black
    # color = (0, 11, 11)
    # # 0
    # color = (225, 7, 7)
    # # 1
    # color = (135, 244, 6)
    # # 2
    color = (0, 0, 255)
    # # 3
    # color = (103, 202, 177)
    # # 4
    # color = (243, 10, 158)
    add_border(file, 8, color)

    # print(file)
