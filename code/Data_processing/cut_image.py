import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import matplotlib.image as mpimg
from Face_detection_cats import *

class DATA_process():
    def __init__(self, image_path,  an_path):
        self.image_path = image_path
   #     self.landmarks = pd.read_csv(an_path, header=None, sep='\t').to_numpy()
        return

    def crop_save_image(self, pathname, file):
        #load image with cv2
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        val = detect_cat_face(img, file)
        if val is None:
            #if no face detected return without saving cropped image
            return
        else:
            a, b, c, d = val[0], val[1], val[2], val[3]
        #crop image
        cropped = img[a:c,b:d]
        #save cropped imahe
        full_path = os.path.join(pathname, file)
        print(file)
        if (cropped.shape[0] == 0) or (cropped.shape[1] == 0):
            print("cropped none in: ", file)
            return
        cv2.imwrite(full_path, cropped)
        return

if __name__=="__main__":
    img_path = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren" \
               r"\Cat_pain_data_for_AI_collaboration\pain_no_pain_data_clinical_population" \
               r"\video_data\Annotated_images_sorted_by_condition"

    dir_img = ["1_hour_after_surgery_worst_pain", "before_surgery_no_pain"]

    an_path =  r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren\Cat_pain_data_for_AI_collaboration" \
           r"\pain_no_pain_data_clinical_population\video_data\Annotated_images_sorted_by_condition"

    save_path_crop = os.path.join(r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren"
                                  r"\Cat_pain_data_for_AI_collaboration\pain_no_pain_data_clinical_population"
                                  r"\video_data\Annotated_images_sorted_by_condition\cropped")
    for dir in dir_img:

        path_dir = os.path.join(img_path, dir)
        #find all files in a directory
        files = os.listdir(path_dir)

        for file in files:
           if ".png" in file:
                img_full_path = os.path.join(path_dir, file)
                data = DATA_process(img_full_path, an_path)
                data.crop_save_image(save_path_crop, "cropped_"+"dir_"+file)





