"""Create an ROC for the image classifier model.

"""
import os
import tensorflow as tf
import pathlib

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def prediction_model():
    """ Creates a ROC plot.

    Opens the saved model. Sets parameters for the height and width for the
    images.
    Creërt ROC plot, laad gemaakte model in en berekent per class
    true positive rate en false positive rate. Geeft een ROC plot
    terug.

    """
    # Loads the saved model.
    # model = tf.keras.models.load_model(
    #     r"C:\Users\eahni\Image-Analysis-Cartoon-Classifier\model_voo"
    #     r"r_augmentatie")
    model = tf.keras.models.load_model(r"C:\Users\eahni\Image-Analysis"
                                       r"-Cartoon-Classifier\model_na"
                                       r"_augmentatie")

    # Parameters for the images.
    img_height = 96
    img_width = 96

    # os.listdir() method in python is used to get the list of all files
    # and directories in the specified directory. If we don’t specify any
    # directory, then list of files and directories in the current working
    # directory will be returned.

    # Creates a list with the names of all the images in the bean directory.
    bean = os.listdir(r"C:\Users\eahni\Image-Analysis-Cartoon-Classi"
                      r"fier\cartoon_backup\data\test\bean")
    # Creates a list with the names of all the images in the conan directory.
    conan = os.listdir(r"C:\Users\eahni\Image-Analysis-Cartoon-Classif"
                       r"ier\cartoon_backup\data\test\conan")
    # Creates a list with the names of all the images in the doraemon
    # directory.
    doraemon = os.listdir(r"C:\Users\eahni\Image-Analysis-Cartoon-Class"
                          r"ifier\cartoon_backup\data\test\doraemon")
    # Creates a list with the names of all the images in the naruto directory.
    naruto = os.listdir(r"C:\Users\eahni\Image-Analysis-Cartoon-Classif"
                        r"ier\cartoon_backup\data\test\naruto")
    # Creates a list with the names of all the images in the shinchan
    # directory.
    shinchan = os.listdir(r"C:\Users\eahni\Image-Analysis-Cartoon-Class"
                          r"ifier\cartoon_backup\data\test\shinchan")

    # Creates two empty lists. One for the true positives and one for the
    # probabilities.
    y_true = []
    y_prob = []

    # Loops over all the images in the list and puts them into arrays.
    # Calculates predictions and scores, these are appended to y_true and
    # y_prob respectively. die worden gebruikt voor de plot.
    for i in bean:
        data_dir_testen_bean = pathlib.Path(r"C:\\Users\\eahni\\"
                                            r"Image-Analysis-Cartoon-"
                                            r"Classifier\\cartoon_backu"
                                            r"p\\data\\test\\bean\\"
                                            + i)
        img = tf.keras.utils.load_img(data_dir_testen_bean,
                                      target_size=(img_height,
                                                   img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        y_true.append([0])
        y_prob.append(score)

    for i in conan:
        data_dir_conan = pathlib.Path(
            r"C:\\Users\\eahni\\Image-Analysis-Cartoon-Classifier\\"
            r"cartoon_backup\\data\\test\\conan\\" + i)
        img = tf.keras.utils.load_img(data_dir_conan,
                                      target_size=(img_height,
                                                   img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        y_true.append([1])
        y_prob.append(score)

    for i in doraemon:
        data_dir_doraemon = pathlib.Path(
            r"C:\\Users\\eahni\\Image-Analysis-Cartoon-Classifier\\"
            r"cartoon_backup\\data\\test\\doraemon\\" + i)
        img = tf.keras.utils.load_img(data_dir_doraemon,
                                      target_size=(img_height,
                                                   img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        y_true.append([2])
        y_prob.append(score)

    for i in naruto:
        data_dir_naruto = pathlib.Path(
            r"C:\\Users\\eahni\\Image-Analysis-Cartoon-Classifier\\"
            r"cartoon_backup\\data\\test\\naruto\\" + i)
        img = tf.keras.utils.load_img(data_dir_naruto,
                                      target_size=(img_height,
                                                   img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        y_true.append([3])
        y_prob.append(score)

    for i in shinchan:
        data_dir_shinchan = pathlib.Path(
            r"C:\\Users\\eahni\\Image-Analysis-Cartoon-Classifier\\"
            r"cartoon_backup\\data\\test\\shinchan\\" + i)
        img = tf.keras.utils.load_img(data_dir_shinchan,
                                      target_size=(img_height,
                                                   img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        y_true.append([4])
        y_prob.append(score)

    y_true = label_binarize(y_true, classes=(0, 1, 2, 3, 4))
    print(y_prob)
    print(y_true)
    auc = roc_auc_score(y_true, y_prob, multi_class='ovr')

    print(auc)


if __name__ == '__main__':
    prediction_model()
