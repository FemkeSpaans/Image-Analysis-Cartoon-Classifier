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
    """

    :return:
    """
    batch_size = 32
    img_height = 96
    img_width = 96

    data_dir_train = pathlib.Path(
        r"C:\Users\eahni\Image-Analysis-Cartoon-Classifier\Train")

    split1 = int(0.8 * len(data_dir_train))
    data_dir_train1 = data_dir_train[:split1]
    data_dir_validatie = data_dir_train[split1:]
    print(data_dir_train1)
    print("xxx")
    print(data_dir_validatie)

    # train = tf.keras.utils.image_dataset_from_directory(
    #     data_dir_train,
    #     seed=123,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size)
    #
    # validatie = tf.keras.utils.image_dataset_from_directory(
    #     data_dir_validatie,
    #     seed=123,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size)
    #
    # class_names = train.class_names
    #
    # return train, validatie, class_names


if __name__ == '__main__':
    datasets()
