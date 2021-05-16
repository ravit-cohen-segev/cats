import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import cv2 as cv

class CatFaceDetector():
    '''
    Class for Cat Face Detection
    '''
    def __init__(self,object_cascade_path):
        '''
        param: object_cascade_path - path for the *.xml defining the parameters for cat face detection algorithm
        source of the haarcascade resource is: https://github.com/opencv/opencv/tree/master/data/haarcascades
        '''

        self.objectCascade=cv.CascadeClassifier(object_cascade_path)


    def detect(self, image, scale_factor,
               min_neighbors,
               min_size):
        '''
        Function return rectangle coordinates of cat face for given image
        param: image - image to process
        param: scale_factor - scale factor used for cat face detection
        param: min_neighbors - minimum number of parameters considered during cat face detection
        param: min_size - minimum size of bounding box for object detected
        '''
        bbox=self.objectCascade.detectMultiScale(image,
                                                scaleFactor=scale_factor,
                                                minNeighbors=min_neighbors,
                                                minSize=min_size)
        return bbox


def detect_cat_face(image, file_name, scale_factor=1.15, min_neighbors=1, min_size=(30,30)):
    '''
    Cat Face detection function
    Identify frontal cat face and display the detected marker over the image
    param: image - the image extracted from the video
    param: scale_factor - scale factor parameter for `detect` function of ObjectDetector object
    param: min_neighbors - min neighbors parameter for `detect` function of ObjectDetector object
    param: min_size - minimum size parameter for f`detect` function of ObjectDetector object
    '''

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    frontal_cascade_path = r"C:\Users\ravit\PycharmProject\cats\cat_face_detection\haarcascade_frontalcatface_extended.xml"

    # Detector for cat frontal face detection created
    fcfd = CatFaceDetector(frontal_cascade_path)

    cat_face = fcfd.detect(image_gray,
                           scale_factor=scale_factor,
                           min_neighbors=min_neighbors,
                           min_size=min_size)

    #if cat_face is empty (cat_face not detected) continue to the next picture
    if cat_face == ():
        print("face not detected in file name:", file_name)
        return

    for x, y, w, h in cat_face:
        # detected cat face shown in color image
        #cv.rectangle(image, (int(x + w / 2), int(y + h / 2)), (int((w + h) / 4)), (0, 127, 255), 3)

        a = x
        b = y
        c = x + w
        d = y + h

        cv.rectangle(image, pt1=(a, b), pt2=(c, d), color=(0, 0, 255), thickness=10)

    # image
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        ax.imshow(image)
    # return image


if __name__=="__main__":

    os.listdir(r"C:\Users\ravit\PycharmProject\cats\data\cat_dataset-kaggle\CAT_00")

    #reading 10 images
    file_list = []
    for root, dirs, files in os.walk(r"C:\Users\ravit\PycharmProject\cats\data\cat_dataset-kaggle\CAT_00"):
        for file in files:
            if file.endswith(".jpg"):
                file_list.append(file)

    excerpt_file_list = file_list[0:20]

    for file in excerpt_file_list:

        img = cv.imread(r"C:/Users/ravit/PycharmProject/cats/data/cat_dataset-kaggle/CAT_00/"+file)
        print(file)
        detect_cat_face(img, file)









