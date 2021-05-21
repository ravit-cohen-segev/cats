import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def count_ext_files(myPath, ext):
    extCounter = 0
    for root, dirs, files in os.walk(myPath):
        for file in files:
            if file.endswith(ext):
                extCounter += 1
    return extCounter

def landmarks_img(file_path, dir, phase):
    os.chdir(file_path)
    idx_list = []
    graph_arrays = []
    # for identifying cat numbers in file name
    # cat_nums_two_digits = [str(i) for i in range(10, 31)]

    #get list of files
    files = os.listdir(file_path)
    for file in files:
      image_path = os.path.join(file_path, file)
      #os.chdir(file_path)
      #upload image
      image = cv2.imread(image_path)
      graph_arrays.append(image)
      #show the image
      load_image.show()
      #keep only cat number
      to_strip = dir.strip('DL')
      new_file_name = file.strip(to_strip + 'cat_')
      #check if 2 digits cat num
      if new_file_name[:2] in cat_nums_two_digits:
          cat_num = new_file_name[:2]
      else:
          cat_num = new_file_name[:1]
          idx_list.append(cat_num)
      #save the condition for subsequent reference
      condition_list = [phase]*len(idx_labels)

      idx_arr = np.vstack((condition_list, idx_list))
    return graph_arrays, idx_arr

def return_split_df(df):
    split_dfs = []
    #get unique values of cat numbers
    uniq_nums = np.unique(df.index, return_index=True)[0].astype('int')
    #convert index type from object to int
    df.index = df.index.astype('int')

    for i, u in enumerate(uniq_nums):
        df_to_append = df[df.index==u].to_numpy()
        split_dfs.append(df_to_append)
    return uniq_nums, split_dfs

def return_sorted_labels(df_nums, labels):
    #find cats that are not in labels
    diff_cats = np.setdiff1d(labels['Cat'].values, df_nums)
    #remove cats from labels
    #first check id diff_cats is not empty
    if diff_cats != np.array([]):
        #get indices of cats needed to be remmoved
        idx_remove = [np.where(labels['Cat'].values == d)[0][0] for d in diff_cats]
        labels = labels.drop(idx_remove, axis=0)

    #return the indices that would sort an array
    sorted_labels = np.zeros(labels.shape)
    i = 0
    for cat in df_nums:
        sorted_labels[i] = labels[labels['Cat']==cat].values
        i += 1
    return sorted_labels

if __name__=="__main__":
    # replace whitespaces in filenames and directories with underscores
    # parent_path = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren\Videos_From_Lauren"

    # replace(parent_path)

    file_path = r"C:\Users\ravit\PycharmProject\cats\code\anot_images_for_DL"

    all_files = []

    #count number of files with .txt extension. This is to make sure that all files were loaded to df"
    count_ext = count_ext_files(file_path,'.txt')

    dir_list = os.listdir(file_path)

    #read df from condition t1
    df_t1 = landmarks_img(os.path.join(file_path, dir_list[1]), dir_list[1], phase='t1')

    # read df from condition t2
    df_t2 = landmarks_img(os.path.join(file_path, dir_list[0]), dir_list[0], phase='t2')






    #return DataFrames split into sub-DataFrames according to animal number
    #cat_nums1, df1_split_list = return_split_df(df1)

    #cat_nums2, df2_split_list = return_split_df(df2)

    #add pain labels from file using cat number and condition
    #excel_file = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren" \
     #            r"\Cat_pain_data_for_AI_collaboration\pain_no_pain_data_-_clinical_population" \
      #           r"\additional_info\botucatu_pain_scoring_tool_results.xls"
    #before surgery (t1):
    #t1_labels = pd.read_excel(excel_file, sheet_name="before surgery - botucatu score")
    #t1_columns_to_drop = np.setdiff1d(t1_labels.columns, ['Cat', 'before surgery - botucatu score '])
    #t1_labels = t1_labels.drop(t1_columns_to_drop, axis=1)
    #drop rows that have cats with no annotations


    #t1_new_labels = return_sorted_labels(cat_nums1,t1_labels)

    # 1 hr after surgery(t2):
    #t2_labels = pd.read_excel(excel_file, sheet_name="1 hr post surg - botucatu score")
    #t2_columns_to_drop = np.setdiff1d(t2_labels.columns, ['Cat', '1hr post surgery - botucatu score '])
    #t2_labels = t2_labels.drop(t2_columns_to_drop, axis=1)

    #t2_new_labels = return_sorted_labels(cat_nums2, t2_labels)

    #combine df1 and df2 and their labels

   # combined_features = [*df1_split_list, *df2_split_list]
   # comnbined_labels = np.vstack((t1_new_labels, t2_new_labels))

   #find out max length of data points x,y per each cat. This to understand the best model dimensions