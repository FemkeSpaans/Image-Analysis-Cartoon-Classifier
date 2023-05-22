import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def datasets():
    """ De functie laad train en validatie set in en welke class namen
    er zijn.

    :return train: specificatie van data
    :return validatie: specificatie van data
    :return class_names: welke class namen er zijn
    """
    # Setting van images
    batch_size = 32
    img_height = 96
    img_width = 96

    # Path naar train data
    data_dir_train = pathlib.Path(
        r"C:\\Users\\eahni\\Image-Analysis-Cartoon-Classifier\\"
        r"cartoon_backup\\data\\train")

    # Path naar validatie data
    data_dir_validatie = pathlib.Path(
        r"C:\\Users\\eahni\\Image-Analysis-Cartoon-Classifier\\"
        r"cartoon_backup\\data\\validation")

    # Batch specificatie maken van train dataset
    train = tf.keras.utils.image_dataset_from_directory(
        data_dir_train,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Batch specificatie maken van validatie dataset
    validatie = tf.keras.utils.image_dataset_from_directory(
        data_dir_validatie,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Alle class namen verkrijgen
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
                                   loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                       from_logits=True),
                                   metrics=['accuracy'])
    model_voor_augmentatie.summary()

    history = model_voor_augmentatie.fit(
        train,
        validation_data=validatie,
        epochs=epochs
    )

    data_augmentation = keras.Sequential(
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

    return t_acc, v_acc, t_loss, v_loss, epochs_range, data_augmentation


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


if __name__ == '__main__':
    train, validatie, class_names = datasets()
    visualisatie_images(train, class_names)
    train, validatie = autotune(train, validatie)
    t_acc, v_acc, t_loss, v_loss, epochs_range, data_augmentation = \
        model_voor_augmentatie(train, validatie, class_names)
    figure_model(t_acc, v_acc, t_loss, v_loss, epochs_range)
