from PIL import Image, ImageOps
import os


def find_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def add_border(image_path, border_size, border_color):
    for img in image_path:
        with Image.open(img) as im:
            new_im = ImageOps.expand(im, border=border_size, fill=border_color)
            new_im.save(img)


if __name__ == '__main__':
    path = './test'  # image folder path
    file = find_files(path)
    add_border(file, 8, (255, 102, 0))

    # print(file)
