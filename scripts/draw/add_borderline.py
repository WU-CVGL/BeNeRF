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
    path = '/Users/pianwan/Downloads/outdoorpool_more/outdoorpool_more/1'  # image folder path
    file = find_files(path)
    # image color: black
    # size: 8
    # color = (0, 11, 11)
    # 0
    # color = (225, 7, 7)
    # # 1
    color = (135, 6, 244)
    # # 2
    # color = (0, 255, 0)
    # # 3
    # color = (103, 177, 202)
    # # 4
    # color = (243, 158, 10)

    # size for livingroom
    # 1 2 3

    # size for outdoorpool
    # 2 3

    # size for whiteroom
    # 2 2
    add_border(file, 3, color)

    # print(file)
