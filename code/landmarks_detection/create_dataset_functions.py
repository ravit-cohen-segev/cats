# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 23:12:16 2021

@author: ravit
"""
import numpy as np
import pandas as pd
import os 
import cv2
import random

class create_dataset_dict():
    def __init__(self,path, image_shape):
        self.path = path
        self.x_cache, self.y_cache = [], []
        self.dim = image_shape


        
    def cats_annotated_in_dir(self, d, train=True, use_data_augmentation=True):
        full_path = os.path.join(self.path,d)
        files = os.listdir(full_path)


        for file in files:
            #check if file is with '.txt' extension. if not than continue to the next file
            if '.txt' not in file:
                continue
          

            #read image
            image_file = file.replace('.txt','.tif')
            image_path = os.path.join(full_path, image_file)
            image_arr = cv2.imread(image_path, 1)
            #if image arr is None continue to the next file
            if image_arr is None:
                continue

            #get width and height of original image for normalizing annotations

            width, height = image_arr.shape[0],  image_arr.shape[1]
            anot_path = os.path.join(full_path, file)
            df_annot = pd.read_csv(anot_path, header=None, sep='\t').T
            annot_arr = df_annot.to_numpy().T

            #remove one file with 49 landmarks
            if len(annot_arr) == 49:
                continue

            #48 is the number of landmarks
            annotation = np.zeros((48, 2), dtype=np.float32)
            annotation[:, 0] = annot_arr[:, 0] / width
            annotation[:, 1] = annot_arr[:, 1] / height
            annotation = np.clip(annotation, 0.0, 1.0)

            #resize image
            resized = cv2.resize(image_arr, self.dim, interpolation=cv2.INTER_LINEAR)

            if train and use_data_augmentation:

                if random.random() >= 0.5:
                    resized = resized[:, ::-1, :]
                    annotation[:, 0] = 1 - annotation[:, 0]
                    annotation[0, :], annotation[1, :] = annotation[1, :], annotation[0, :].copy()
                    annotation[3:6, :], annotation[6:9, :] = annotation[6:9, :], annotation[3:6, :].copy()
                # PCA Color Augmentation
                img_array = self.pca_color_augmentation(resized)

            self.x_cache.append(img_array)
            self.y_cache.append(annotation)
            #move to the next index/key

        return

    def pca_color_augmentation(self, image_array_input):
        assert image_array_input.ndim == 3 and image_array_input.shape[2] == 3
        assert image_array_input.dtype == np.uint8

        img = image_array_input.reshape(-1, 3).astype(np.float32)
        img = (img - np.mean(img, axis=0)) / np.std(img, axis=0)

        cov = np.cov(img, rowvar=False)
        lambd_eigen_value, p_eigen_vector = np.linalg.eig(cov)

        rand = np.random.randn(3) * 0.1
        delta = np.dot(p_eigen_vector, rand*lambd_eigen_value)
        delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]

        img_out = np.clip(image_array_input + delta, 0, 255).astype(np.uint8)
        return img_out

    
    def create_dict(self):
        #create dir_list from directory
        dir_list = os.listdir(self.path)
        for d in dir_list:
            self.cats_annotated_in_dir(d)
        return self.x_cache, self.y_cache
            
          
if __name__=="__main__":
    dir_path_clinical = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren\Cat_pain_data_for_AI_collaboration\pain_no_pain_data_clinical_population\video_data\Annotated_images_sorted_by_condition"
    dir_path_div_pain = r"C:/Users/ravit/PycharmProject/cats/Laurens_dataset/Videos_From_Lauren/Cat_pain_data_for_AI_collaboration/pain_no_pain_data_diverse_population_/pain"
    dir_path_div_no_pain = r"C:/Users/ravit/PycharmProject/cats/Laurens_dataset/Videos_From_Lauren/Cat_pain_data_for_AI_collaboration/pain_no_pain_data_diverse_population_/no_pain"
    
    create_data = create_dataset_dict(dir_path_clinical, 0, (224,224))
    dataset_dict = create_data.create_dict()

    create_data_div_pain = create_dataset_dict(dir_path_div_pain, len(dataset_dict), (224,224))
    dataset_div_pain = create_data_div_pain.create_dict()

    create_data_div_no_pain = create_dataset_dict(dir_path_div_no_pain, len(dataset_dict)+len(dataset_div_pain), (224,224))
    dataset_div_no_pain = create_data_div_no_pain.create_dict()

    dataset_dict.update(dataset_div_pain)
    dataset_dict.update(dataset_div_no_pain)