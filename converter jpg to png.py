from PIL import Image
import os


def filedirectory(filedir):
    """

    :param filedir:
    :return:
    """
    filelist = []
    for filename in os.listdir(filedir):
        f = os.path.join(filedir, filename)
        filelist.append(f)
    return filelist


def convert_image(filedirectory):
    """

    :param filedirectory:
    :return:
    """
    im = Image.open(filedirectory)
    namereplacement = filedirectory.replace(".jpg", "")
    im.save(namereplacement + ".png", "PNG")


def main():
    allfilelists = []
    fileswithpicturepath = []
    # Kijkt naar de eerste volgende path (test, train, validatie)
    filelist = filedirectory("cartoon_backup\\data")
    print("x")

    # Kijkt naar de eerst volgende path (bean, conan, etc.)
    for i in filelist:
        filelist = filedirectory(i)
        allfilelists.append(filelist)
    print("xx")

    # Kijkt naar images in de hele path (1.jpg, 2.jpg, 3.jpg, etc...)
    for i in allfilelists:
        for i2 in i:
            filewithpicturelist = filedirectory(i2)
            fileswithpicturepath.append(filewithpicturelist)
    print("xxx")

    # Pakt de hele path van de images
    for dir in fileswithpicturepath:
        for picture in dir:
            # Zet de images om van jpg naar png
            convert_image(picture)
            # verwijderd de jpg image
            os.remove(picture)
    print("xxxx")

if __name__ == '__main__':
    main()
