""" Create an image classifier for cartoon images.

The data is separate into three directories; test, train, and
validation. In each of these directories are five directories; bean,
conan, doraemon,naruto, and shinchan. In these directories are image
files of the
cartoon.In total there are 2468 images files, they are divided as
follows:

test directory (247 image files, 10% of the entire data):
- bean = 49 image files.
- conan = 50 image files.
- doraemon = 49 image files.
- naruto = 48 image files.
- shinchan = 51 image files

train directory (1974 image files, 80% of the entire data):
- bean = 296 image files.
- conan = 499 image files.
- doraemon = 482 image files.
- naruto = 202 image files.
- shinchan = 495 image files

validation directory (247 image files, 10% of the entire data):
- bean = 27 image files.
- conan = 69 image files.
- doraemon = 54 image files.
- naruto = 15 image files.
- shinchan = 82 image files
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

    Sets three parameters for the images, which will be used in the
    datasets; batch_size, img_height, and img_width. The batch size is a
    number of samples which are processed before the model is updated,
    and moves on to the next batch. Image height and image width are
    used to resize the images, so all the images are the same size.

    Saves the paths to two separate directories, which contain the
    training data and the validation data.

    Generates two separate datasets from the files in the directories
    which were saved in the paths; train and validation. Uses the
    earlier specified parameters for batch size, image height, and
    image width.

    Retrieves the class names from the training dataset. Class names
    correspond to the name of the directory.
    Class names = bean, conan, doraemon, naruto, shinchan.

    :return train: train dataset.
    :return validation: validation dataset.
    :return class_names: class names of the directories from the train
    data.
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
    data_dir_validation = pathlib.Path(
        r"C:\\Users\\eahni\\Image-Analysis-Cartoon-Classifier\\"
        r"cartoon_backup\\data\\validation")

    # Generates a dataset from the files.
    train = tf.keras.utils.image_dataset_from_directory(
        data_dir_train,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Generates a dataset from the files.
    validation = tf.keras.utils.image_dataset_from_directory(
        data_dir_validation,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Class names which correspond to the name of the directory.
    class_names = train.class_names

    return train, validation, class_names


def visualisation_images(train, class_names):
    """Shows images from the training dataset.

    Sets the sizes of the images to a height and width of 10.

    Uses a for loop to loop over the first 9 images in the training
    dataset. Then shows these images with the label of which class they
    belong to.


    :param train: train dataset.
    :param class_names: class names of the directories from the train
    data.
    """
    plt.figure(figsize=(10, 10))
    for images, labels in train.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.show()


def autotune(train, validation):
    """Optimizing the performance of the datasets.

    Autotune sets tf.data.AUTOTUNE, this prompts the tf.data runtime to
    tune the value dynamically at runtime. Whilst the model is executing
    the training step, the input pipeline is reading the data for the
    next training step. This will reduce the step time of the training
    and the time it takes to extract the data.

    Next on both the training and the validation dataset it uses.cache()
    and .prefetch(). cache() stores the images in memory after they have
    been loaded off the disk during the first epoch. By loading the
    images in memory they do not have to be loaded every new epoch,
    reducing the runtime. prefetch() will load the next batch whilst
    the model is executing the training step of the previous batch.

    :param train: train dataset.
    :param validatie: validation dataset.
    :return: autotune_train: train dataset after autotune.
    :return: autotune_validatie: validation dataset after autotune.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    autotune_train = train.cache().shuffle(1000).prefetch(
        buffer_size=AUTOTUNE)
    autotune_validatie = validation.cache().prefetch(
        buffer_size=AUTOTUNE)

    return autotune_train, autotune_validatie


def model_before_augmentation(train, validation, class_names):
    """Creates a model.

    Sets two parameters for the images, which will be used in the
    datasets; img_height and img_width. Image height and image width are
    used to resize the images, so all the images are the same size. Sets
    a parameter for the number of epochs. An epoch refers to one cycle
    through the full training dataset. Counts the number of class names.

    A Sequential model builds every layer sequentially, each layer has
    exactly one input tensor and one output tensor. The data goes
    through every layer from top to bottom until the data has reached
    the end of the model. The layers in the model are; Rescaling,
    Conv2D, MaxPooling2D, Flatten, and Dense. The Rescaling layer is a
    preprocessing layer which rescales the input values to a new range.
    The Conv2D layer is a 2D convolution layer, it performs spatial
    convolution over the images. It takes a part of the image and
    compares them to neighboring parts. The MaxPooling2D layer down
    samples the output from the previous layer without losing the
    important features. The Flatten layer puts the input matrix from the
    previous layer and puts it into a single array, hereby flatting the
    input. The Dense layer is deeply connected with the previous layer,
    meaning each neuron in the dense layer receives input from all the
    neurons of the previous layer.

    After the model is build it is compiled before the training starts.
    During compilation, it checks for format errors, defines the loss
    function, learning rate, and other metrics. Next a summary of the
    model is given.

    After the previous steps are done the training of the model begins.
    The parameters for the training accuracy, validation accuracy,
    training loss, and validation loss are saved. Lastly the model is
    saved.

    To reduce the change of overfitting data augmentation is done with
    the model after the training has taken place. Data augmentation
    consists of three preprocessing layers; RandomFlip, RandomRotation,
    and RandomZoom. RandomFlip randomly flips the images during the
    training. RandomRotation randomly rotates images during the
    training. RandomZoom randomly zooms in on the images during the
    training. This steps is done during this function
    because it needs the model which is created during this function.

    :param train: train dataset.
    :param validatie: validation dataset.
    :param class_names: class names of the directories from the train
    data.
    :return t_acc: training accuracy.
    :return v_acc: validation accuracy.
    :return t_loss: training loss.
    :return v_loss: validation loss.
    :return epochs_range: range of the epochs.
    :return data_aug: data after data augmentation.
    """

    # Parameters for the images.
    img_height = 96
    img_width = 96
    # Parameter for number of epochs (iterations).
    epochs = 10
    # Counts the number of class names.
    num_classes = len(class_names)
    # Create model.
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
    # Compile model.
    model_voor_augmentatie.compile(optimizer='adam',
                                   loss=tf.keras.losses.
                                   SparseCategoricalCrossentropy
                                   (from_logits=True),
                                   metrics=['accuracy'])
    # Summary of the model.
    model_voor_augmentatie.summary()
    # Training of the model.
    history = model_voor_augmentatie.fit(
        train,
        validation_data=validation,
        epochs=epochs
    )
    # Data augmentation.
    data_aug = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height, img_width, 3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )
    # Saving the parameters of the train and validation sets for
    # visualisation.
    t_acc = history.history['accuracy']
    v_acc = history.history['val_accuracy']
    t_loss = history.history['loss']
    v_loss = history.history['val_loss']
    epochs_range = range(epochs)
    # Saving the model.
    model_voor_augmentatie.save(r"C:\\Users\\eahni\\"
                                r"Image-Analysis-Cartoon-Classifier\\"
                                r"model_voor_augmentatie")

    return t_acc, v_acc, t_loss, v_loss, epochs_range, data_aug


def figure_model(t_acc, v_acc, t_loss, v_loss, epochs_range):
    """Create figures of the accuracy and loss.

    Creates two figures, on of the training and validation accuracy, and
    one of the training and validation loss.

    :param epochs_range: range of the epochs.
    :param t_acc: training accuracy.
    :param v_acc: validation accuracy.
    :param t_loss: training loss.
    :param v_loss: validation loss.
    """
    # Set the size of the figure.
    plt.figure(figsize=(8, 8))
    # Creates a figure and a grid of subplots.
    plt.subplot(1, 2, 1)
    # Plots the training accuracy.
    plt.plot(epochs_range, t_acc, label='Training Accuracy')
    # Plots the validation accuracy.
    plt.plot(epochs_range, v_acc, label='Validation Accuracy')
    # Creates a legend.
    plt.legend(loc='lower right')
    # Creates a title.
    plt.title('Training and Validation Accuracy')
    # Creates a label for the X axis.
    plt.xlabel("Number of Epochs")
    # Creates a label for the Y axis.
    plt.ylabel("Accuracy")

    # Creates a figure and a grid of subplots.
    plt.subplot(1, 2, 2)
    # Plots the training loss.
    plt.plot(epochs_range, t_loss, label='Training Loss')
    # Plots the validation loss.
    plt.plot(epochs_range, v_loss, label='Validation Loss')
    # Creates a legend.
    plt.legend(loc='upper right')
    # Creates a title.
    plt.title('Training and Validation Loss')
    # Creates a label for the X axis.
    plt.xlabel("Number of Epochs")
    # Creates a label for the Y axis.
    plt.ylabel("Loss")
    # Shows figures.
    plt.show()


def augmentation_figure(train, data_aug):
    """Creates figures of the data after augmentation.

    Creates a figure with 9 images from the train dataset and shows these.

    :param train: train dataset.
    :param data_aug: data after data augmentation.
    """
    # Set size for the figures.
    plt.figure(figsize=(10, 10))
    # Loops over the training set.
    for images, _ in train.take(1):
        # Picks 9 images.
        for i in range(9):
            augmented_images = data_aug(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")


def model_aug(data_aug, class_names, train, validation):
    """Creates a model with the augmented data.

    A Sequential model builds every layer sequentially, each layer has
    exactly one input tensor and one output tensor. The data goes
    through every layer from top to bottom until the data has reached
    the end of the model. The layers in the model are; Rescaling,
    Conv2D, MaxPooling2D, Flatten, and Dense. The Rescaling layer is a
    preprocessing layer which rescales the input values to a new range.
    The Conv2D layer is a 2D convolution layer, it performs spatial
    convolution over the images. It takes a part of the image and
    compares them to neighboring parts. The MaxPooling2D layer down
    samples the output from the previous layer without losing the
    important features. The Flatten layer puts the input matrix from the
    previous layer and puts it into a single array, hereby flatting the
    input. The Dense layer is deeply connected with the previous layer,
    meaning each neuron in the dense layer receives input from all the
    neurons of the previous layer.

    After the model is build it is compiled before the training starts.
    During compilation, it checks for format errors, defines the loss
    function, learning rate, and other metrics. Next a summary of the
    model is given.

    After the previous steps are done the training of the model begins.
    The parameters for the training accuracy, validation accuracy,
    training loss, and validation loss are saved. Lastly the model is
    saved.

    :param data_aug: data after data augmentation.
    :param class_names: class names of the directories from the train
    data.
    :param train: train dataset.
    :param validatie: validation dataset.
    :return history: trained model.
    :return epochs: Parameter for number of epochs (iterations).
    """

    # Counts the number of class names.
    num_classes = len(class_names)
    # Create model.
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
    # Compile model.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    # Summary of model.
    model.summary()
    # Parameter for number of epochs (iterations).
    epochs = 15
    # Training of the model.
    history = model.fit(
        train,
        validation_data=validation,
        epochs=epochs
    )
    # Save model.
    model.save(r"C:\Users\eahni\Image-Analysis-Cartoon-Classifier\
    model_na_augmentatie")

    return history, epochs


def figure_aug(history, epochs):
    """Create figures of the accuracy and loss of the augmented data.

    Creates two figures, on of the training and validation accuracy, and
    one of the training and validation loss of the augmented data. This is
    done using a history object. A history object is a record of the training
    and validation loss values and metrics values at the successive epochs.

    :param history: trained model.
    :param epochs: Parameter for number of epochs (iterations).
    """

    # History object of accuracy.
    acc = history.history['accuracy']
    # History object of validation accuracy.
    val_acc = history.history['val_accuracy']
    # History object of loss.
    loss = history.history['loss']
    # History object of validation loss.
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)
    # Set the size of the figure.
    plt.figure(figsize=(8, 8))
    # Creates a figure and a grid of subplots.
    plt.subplot(1, 2, 1)
    # Plots the training accuracy.
    plt.plot(epochs_range, acc, label='Training Accuracy')
    # Plots the validation accuracy.
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # Creates a legend.
    plt.legend(loc='lower right')
    # Creates a title.
    plt.title('Training and Validation Accuracy')
    # Creates a label for the X axis.
    plt.xlabel("Number of Epochs")
    # Creates a label for the Y axis.
    plt.ylabel("Accuracy")

    # Creates a figure and a grid of subplots.
    plt.subplot(1, 2, 2)
    # Plots the training loss.
    plt.plot(epochs_range, loss, label='Training Loss')
    # Plots the validation loss.
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    # Creates a legend.
    plt.legend(loc='upper right')
    # Creates a title.
    plt.title('Training and Validation Loss')
    # Creates a label for the X axis.
    plt.xlabel("Number of Epochs")
    # Creates a label for the Y axis.
    plt.ylabel("Accuracy")
    # Shows figures.
    plt.show()

if __name__ == '__main__':
    train, validation, class_names = datasets()
    visualisation_images(train, class_names)
    train, validation = autotune(train, validation)
    t_acc, v_acc, t_loss, v_loss, epochs_range, data_aug = \
        model_before_augmentation(train, validation, class_names)
    figure_model(t_acc, v_acc, t_loss, v_loss, epochs_range)
    augmentation_figure(train, data_aug)
    history, epochs = model_aug(data_aug, class_names, train, validation)
    figure_aug(history, epochs)
