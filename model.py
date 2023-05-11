import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import h5py


def datasets():
    """ De functie

    :return:
    """
    batch_size = 32
    img_height = 96
    img_width = 96

    data_dir_train = pathlib.Path(
        r"C:\\Users\\eahni\\Image-Analysis-Cartoon-Classifier\\"
        r"cartoon_backup\\data\\train")

    data_dir_validatie = pathlib.Path(
        r"C:\\Users\\eahni\\Image-Analysis-Cartoon-Classifier\\"
        r"cartoon_backup\\data\\validation")

    train = tf.keras.utils.image_dataset_from_directory(
        data_dir_train,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    validatie = tf.keras.utils.image_dataset_from_directory(
        data_dir_validatie,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train.class_names

    return train, validatie, class_names


if __name__ == '__main__':
    train, validatie, class_names = datasets()
