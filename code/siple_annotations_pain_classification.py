import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

class DATA_landmarks():
    def __init__(self, dir_path):
        self.path = dir_path
        return

    def check_cat_name_in_file(self,file):
        #create a list of cats with 2 digits and not 1
        cat_nums_two_digits = [str(i) for i in range(10, 31)]
        #remove first strings from name that are not digits
        for i,s in enumerate(file):
            if str.isdigit(s):
                new_file = file[i:]
                break

        # check if 2 digits cat num
        if new_file[:2] in cat_nums_two_digits:
            cat_num = new_file[:2]
        else:
            cat_num = new_file[:1]
        return cat_num

    def calc_euclidean(self, c1, c2):
        return np.sqrt( (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 )

    def calc_euc_vec(self, arr, feature_num=48):
        #list of euclidean distance from the upper triangle
        li_distance = []
        #calculate the upper triangle in euclidean distance matrix
        for i in range(feature_num):
            for j in range(i+1, feature_num):
                euc = self.calc_euclidean(arr[:,i], arr[:,j])
                li_distance.append(euc)
        return np.array(li_distance)

    def create_array_landmarks(self, dir_list, feature_num=48):
        #calc vec length
        #count = 0
        #for i in range(1,feature_num):
         #   count += i
          # count is 1128
        columns = ['cat', 'label']
        labels_df = pd.DataFrame(columns = columns)
        euc_arr = np.empty((0,1128))
        idx_count = 0
        for i, dir in enumerate(dir_list):
            full_path = os.path.join(self.path, dir)
            all_files = os.listdir(full_path)
            for file in all_files:
                #check if file is with '.txt' extension. if not than continue to the next file
                if '.txt' not in file:
                    continue

                anot_path = os.path.join(full_path,file)
                #read annotations
                df_annot = pd.read_csv(anot_path, header=None, sep='\t').T
                #add row with cat num and pain clasification
                # add label, if t1 than no pain (0) if t2 than pain (1)
                labels_df.loc[idx_count] = [self.check_cat_name_in_file(file), i]
                #update idx_count
                idx_count += 1
                #convert to numpy
                arr_from_df = df_annot.to_numpy()
                euc_vec = self.calc_euc_vec(arr_from_df)
                euc_arr = np.vstack((euc_arr, euc_vec))
                #normalize data by max value in each row
                max = np.max(euc_arr)
                normalized_euc = euc_arr / max
        return normalized_euc, labels_df

if __name__=="__main__":
    dir_path = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren" \
               r"\Cat_pain_data_for_AI_collaboration\pain_no_pain_data_clinical_population" \
               r"\video_data\Annotated_images_sorted_by_condition"

    dir_list = ['1_hour_after_surgery_worst_pain', 'before_surgery_no_pain']

    DATA_build = DATA_landmarks(dir_path)
    landmarks_euc_array, labels = DATA_build.create_array_landmarks(dir_list)


