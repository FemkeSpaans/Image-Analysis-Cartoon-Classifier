import os
import tensorflow as tf
import pathlib

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def prediction_model():
    """ Creërt ROC plot, laad gemaakte model in en berekent per class
    true positive rate en false positive rate. Geeft een ROC plot
    terug.

    """
    # Laad het gemaakte model in
    # model = tf.keras.models.load_model(
    #     r"C:\Users\eahni\Image-Analysis-Cartoon-Classifier\model_voo"
    #     r"r_augmentatie")
    model = tf.keras.models.load_model(r"C:\Users\eahni\Image-Analysis"
                                       r"-Cartoon-Classifier\model_na"
                                       r"_augmentatie")

    # Zet de hoogte en breedte voor image
    img_height = 96
    img_width = 96

    # Maak een lijst met alle namen van images in map
    bean = os.listdir(r"C:\Users\eahni\Image-Analysis-Cartoon-Classi"
                      r"fier\cartoon_backup\data\test\bean")
    conan = os.listdir(r"C:\Users\eahni\Image-Analysis-Cartoon-Classif"
                       r"ier\cartoon_backup\data\test\conan")
    doraemon = os.listdir(r"C:\Users\eahni\Image-Analysis-Cartoon-Class"
                          r"ifier\cartoon_backup\data\test\doraemon")
    naruto = os.listdir(r"C:\Users\eahni\Image-Analysis-Cartoon-Classif"
                        r"ier\cartoon_backup\data\test\naruto")
    shinchan = os.listdir(r"C:\Users\eahni\Image-Analysis-Cartoon-Class"
                          r"ifier\cartoon_backup\data\test\shinchan")

    # Maak 2 lege lijsten aan voor true positives en kansen
    y_true = []
    y_prob = []

    # Voor alle images in de lijst worden images ingeladen en array's
    # gemaakt. Vervolgens worden voorspellingen gedaan en score berekend
    # Deze waardes worden toegevoegd aan de 2 lijsten, die worden
    # gebruikt voor de plot.
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
