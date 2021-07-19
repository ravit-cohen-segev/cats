# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 23:12:16 2021

@author: ravit
"""
import numpy as np
import pandas as pd
import os 
import cv2

class create_dataset_dict():
    def __init__(self,path, i):
        self.path = path
        self.annotated = {}
        self.i = i
        
    def cats_annotated_in_dir(self, d):
        full_path = os.path.join(self.path,d)
        files = os.listdir(full_path)
        for file in files:
            #check if file is with '.txt' extension. if not than continue to the next file
            if '.txt' not in file:
                continue
          
            anot_path = os.path.join(full_path,file)        
            df_annot = pd.read_csv(anot_path, header=None, sep='\t').T
            annot_arr = df_annot.to_numpy()
            #read image
            image_file = file.replace('.txt','.tif')
            image_path = os.path.join(full_path, image_file)
            image_arr = cv2.imread(image_path, 1)
            self.annotated[str(self.i)] = [image_arr, annot_arr]
            #move to the next index/key
            self.i += 1
        
        return 
    
    def create_dict(self):
        #create dir_list from directory
        dir_list = os.listdir(self.path)
        for d in dir_list:
            self.cats_annotated_in_dir(d)
        return self.annotated
            
          
if __name__=="__main__":
    dir_path_clinical = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren\Cat_pain_data_for_AI_collaboration\pain_no_pain_data_clinical_population\video_data\Annotated_images_sorted_by_condition"
    dir_path_div_pain = r"C:/Users/ravit/PycharmProject/cats/Laurens_dataset/Videos_From_Lauren/Cat_pain_data_for_AI_collaboration/pain_no_pain_data_diverse_population_/pain"
    dir_path_div_no_pain = r"C:/Users/ravit/PycharmProject/cats/Laurens_dataset/Videos_From_Lauren/Cat_pain_data_for_AI_collaboration/pain_no_pain_data_diverse_population_/no_pain"
    
    create_data = create_dataset_dict(dir_path_clinical, 0)
    dataset_dict = create_data.create_dict()

    create_data_div_pain = create_dataset_dict(dir_path_div_pain, len(dataset_dict))
    dataset_div_pain = create_data_div_pain.create_dict()

    create_data_div_no_pain = create_dataset_dict(dir_path_div_no_pain, len(dataset_dict)+len(dataset_div_pain))
    dataset_div_no_pain = create_data_div_no_pain.create_dict()

    dataset_dict.update(dataset_div_pain)
    dataset_dict.update(dataset_div_no_pain)