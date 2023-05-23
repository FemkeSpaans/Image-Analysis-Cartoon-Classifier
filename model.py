""" Create an image classifier for cartoon images.

"""
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def datasets():
    """ Opens files and puts them into datasets.

    Sets three parameters for the images, which will be used in the datasets;
    batch_size, img_height, and img_width. The batch size is a number of
    samples which are processed before the model is updated, and moves on to
    the next batch. Image height and image width are used to resize the images,
    so all the images are the same size.

    Saves the paths to two separate directories, which contain the training
    data and the validation data.

    Generates two separate datasets from the files in the directories which
    were saved in the paths; train and validation. Uses the earlier specified
    parameters for batch size, image height, and image width.

    Retrieves the class names from the training dataset. Class names correspond
    to the name of the directory.
    Class names = bean, conan, doraemon, naruto, shinchan.

    :return train: training dataset.
    :return validatie: validation dataset.
    :return class_names: class names of the directories from the train data.
    """

    # Parameters for the images.
    batch_size = 32
    img_height = 96
    img_width = 96

    # Path to the training data.
    data_dir_train = pathlib.Path(
        r"C:\\Users\\eahni\\Image-Analysis-Cartoon-Classifier\\"
        r"cartoon_backup\\data\\train")

    # Path to the validation data.
    data_dir_validatie = pathlib.Path(
        r"C:\\Users\\eahni\\Image-Analysis-Cartoon-Classifier\\"
        r"cartoon_backup\\data\\validation")

    # Generates a dataset from the files.
    train = tf.keras.utils.image_dataset_from_directory(
        data_dir_train,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Generates a dataset from the files.
    validatie = tf.keras.utils.image_dataset_from_directory(
        data_dir_validatie,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Class names which correspond to the name of the directory.
    class_names = train.class_names

    return train, validatie, class_names


def visualisatie_images(train, class_names):
    """ Visualisatie van images met de daarbij behorende class namen

    """
    plt.figure(figsize=(10, 10))
    for images, labels in train.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.show()


def autotune(train, validatie):
    """

    :return:
    """
    AUTOTUNE = tf.data.AUTOTUNE

    autotune_train = train.cache().shuffle(1000).prefetch(
        buffer_size=AUTOTUNE)
    autotune_validatie = validatie.cache().prefetch(
        buffer_size=AUTOTUNE)

    return autotune_train, autotune_validatie


def model_voor_augmentatie(train, validatie, class_names):
    """

    :return:
    """
    img_height = 96
    img_width = 96
    epochs = 10
    num_classes = len(class_names)

    model_voor_augmentatie = Sequential([
        layers.Rescaling(1. / 255,
                         input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model_voor_augmentatie.compile(optimizer='adam',
                                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                   metrics=['accuracy'])
    model_voor_augmentatie.summary()

    history = model_voor_augmentatie.fit(
        train,
        validation_data=validatie,
        epochs=epochs
    )

    data_aug = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height, img_width, 3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    t_acc = history.history['accuracy']
    v_acc = history.history['val_accuracy']
    t_loss = history.history['loss']
    v_loss = history.history['val_loss']
    epochs_range = range(epochs)

    model_voor_augmentatie.save(r"C:\\Users\\eahni\\"
                                r"Image-Analysis-Cartoon-Classifier\\"
                                r"model_voor_augmentatie")

    return t_acc, v_acc, t_loss, v_loss, epochs_range, data_aug


def figure_model(t_acc, v_acc, t_loss, v_loss, epochs_range):
    """

    :param epochs_range:
    :param t_acc:
    :param v_acc:
    :param t_loss:
    :param v_loss:
    :return:
    """
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, t_acc, label='Training Accuracy')
    plt.plot(epochs_range, v_acc, label='Validatie Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training en Validatie Accuracy')
    plt.xlabel("Aantal epochs")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, t_loss, label='Training Loss')
    plt.plot(epochs_range, v_loss, label='Validatie Loss')
    plt.legend(loc='upper right')
    plt.title('Training en Validatie Loss')
    plt.xlabel("Aantal epochs")
    plt.ylabel("Loss")
    plt.show()


def augmentation_figure(trainset, data_aug):
    """

    :param trainset:
    :param data_aug:
    :return:
    """
    plt.figure(figsize=(10, 10))
    for images, _ in trainset.take(1):
        for i in range(9):
            augmented_images = data_aug(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")


def model_aug(data_aug, class_names, trainset, validatieset):
    """

    :return:
    """
    num_classes = len(class_names)
    model = Sequential([
        data_aug,
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs = 15
    history = model.fit(
        trainset,
        validation_data=validatieset,
        epochs=epochs
    )

    model.save(r"C:\Users\eahni\Image-Analysis-Cartoon-Classifier\model_na_augmentatie")

    return history, epochs


def figure_aug(history, epochs):
    """

    :param history:
    :return:
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel("Aantal epochs")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel("Aantal epochs")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == '__main__':
    train, validatie, class_names = datasets()
    visualisatie_images(train, class_names)
    train, validatie = autotune(train, validatie)
    t_acc, v_acc, t_loss, v_loss, epochs_range, data_aug = \
        model_voor_augmentatie(train, validatie, class_names)
    figure_model(t_acc, v_acc, t_loss, v_loss, epochs_range)
    augmentation_figure(train, data_aug)
    history, epochs = model_aug(data_aug, class_names, train, validatie)
    figure_aug(history, epochs)
