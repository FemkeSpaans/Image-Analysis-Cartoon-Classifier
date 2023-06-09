"""Converts image files saved in a jpg format to a png format.
"""

from PIL import Image
import os


def file_directory(file_dir):
    """ Creates a list containing the images files from a directory.

    :param file_dir: contains 5 lists of the directories with the image files.
    :return file_list: list containing the directories and the respective
    images.
    """
    file_list = []
    for file_name in os.listdir(file_dir):
        f = os.path.join(file_dir, file_name)
        file_list.append(f)
    return file_list


def convert_image(file_directory):
    """Replaces the .jpg format to .png format.

    :param file_directory: list containing the paths of the directories.
    """
    im = Image.open(file_directory)
    name_replacement = file_directory.replace(".jpg", "")
    im.save(name_replacement + ".png", "PNG")


def main():
    all_file_lists = []
    files_with_picture_path = []
    # Finds the next directory in the path (test, train, validation).
    file_list = file_directory("cartoon_backup\\data")
    print("x")

    # Finds the next directory in the path (bean, conan, etc.).
    for i in file_list:
        file_list = file_directory(i)
        all_file_lists.append(file_list)
    print("xx")

    # Finds the next image in the path (1.jpg, 2.jpg, 3.jpg, etc...).
    for i in all_file_lists:
        for i2 in i:
            file_with_picture_list = file_directory(i2)
            files_with_picture_path.append(file_with_picture_list)
    print("xxx")

    # Entire path to the images.
    for directory in files_with_picture_path:
        for picture in directory:
            # Converts the images from .jpg to .png.
            convert_image(picture)
            # Deletes the .jpg image files.
            os.remove(picture)
    print("xxxx")

if __name__ == '__main__':
    main()
