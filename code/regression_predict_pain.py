import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def count_ext_files(myPath, ext):
    extCounter = 0
    for root, dirs, files in os.walk(myPath):
        for file in files:
            if file.endswith(ext):
                extCounter += 1
    return extCounter

def replace(parent): #to replace whitespaces with undescores in file and dir names
    for path, folders, files in os.walk(parent):
        for i in range(len(folders)):
            new_name = folders[i].replace(' ', '_')
            os.rename(os.path.join(path, folders[i]), os.path.join(path, new_name))
            folders[i] = new_name


        for f in files:
            os.rename(os.path.join(path, f), os.path.join(path, f.replace(' ', '_')))

def return_landmarks_df(file_path, dir, phase):
    # create dataframe containing all annotated landmarks-48 features together with cat number
    # create a list of column names
    columns = [str(i) for i in range(1, 49)]

    df_landmarks = pd.DataFrame(columns=columns)
    os.chdir(file_path)

    for root, dirs, files in os.walk(file_path, topdown=False):
        #for identifying cat numbers in file name
        cat_nums_two_digits = [str(i) for i in range(10,31)]

        #get list of files
        #file_names = os.listdir(dir_path)

        for file in files:
            if '.txt' in file:
                #extract features from file and transpose table

                full_path = os.path.join(file_path, file)
                small_df = pd.read_csv(full_path, header=None, sep='\t').T
                #cat 14 video 1.8 is with 49 features, remove from analysis
                if small_df.shape[1]>48:
                    continue
                small_df.columns = columns
                #remove 'cat_' from file name
                if '_cat' in file:
                    file = file.replace('_cat_','')
                if 'cat' in file:
                    file = file.replace('cat_', '')

                #in case the cat number is with two digits
                if file[:2] in cat_nums_two_digits:
                    cat_num = [file[:2], file[:2]]
                else:
                    cat_num = [file[:1], file[:1]]

                small_df['cat_num'] = cat_num
                #save the condition in the last column
                small_df['coordinates'] = ('x', 'y')
                small_df['condition'] = [phase]*2
                small_df = small_df.set_index('cat_num')

                df_landmarks = df_landmarks.append(small_df)

        return df_landmarks

def return_split_df(df):
    split_dfs = []
    #get unique values of cat numbers
    uniq_nums = np.unique(df.index, return_index=True)
    for i, u in enumerate(uniq_nums[0]):
        df_to_append = df[df.index==u].to_numpy()
        split_dfs.append(df_to_append)
    return uniq_nums, split_dfs

def return_sorted_labels(df_nums, labels):
    #find cats that are not in labels
    diff_cats = np.setdiff1d(labels['Cat'].values, df_nums[0].astype('int'))
    #remove cats from labels
    #first check id diff_cats is not empty
    if diff_cats != np.array([]):
        #get indices of cats needed to be remmoved
        idx_remove = [np.where(labels['Cat'].values == d)[0][0] for d in diff_cats]
        labels = labels.drop(idx_remove, axis=0)

    #return the indices that would sort an array
    sorted_idx = np.argsort(df_nums[1])
    #convert to numpy array
    labels_arr = labels.to_numpy()
    sorted_labels = labels_arr[sorted_idx]
    return sorted_labels
if __name__=="__main__":
    # replace whitespaces in filenames and directories with underscores
    # parent_path = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren\Videos_From_Lauren"

    # replace(parent_path)

    file_path = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren" \
                r"\Cat_pain_data_for_AI_collaboration\pain_no_pain_data_-_clinical_population" \
                r"\video_data\Annotated_images_sorted_by_condition_"

    all_files = []

    #count number of files with .txt extension. This is to make sure that all files were loaded to df"
    count_ext = count_ext_files(file_path,'.txt')

    dir_list = os.listdir(file_path)

    df1 = return_landmarks_df(os.path.join(file_path, dir_list[3]), dir_list[3], phase='t1')

    df2 = return_landmarks_df(os.path.join(file_path, dir_list[0]), dir_list[0], phase='t2')

    #return DataFrames split into sub-DataFrames according to animal number
    cat_nums1, df1_split_list = return_split_df(df1)

    cat_nums2, df2_split_list = return_split_df(df2)

    #add pain labels from file using cat number and condition
    excel_file = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren" \
                 r"\Cat_pain_data_for_AI_collaboration\pain_no_pain_data_-_clinical_population" \
                 r"\additional_info\botucatu_pain_scoring_tool_results.xls"
    #before surgery (t1):
    t1_labels = pd.read_excel(excel_file, sheet_name="before surgery - botucatu score")
    t1_columns_to_drop = np.setdiff1d(t1_labels.columns, ['Cat', 'before surgery - botucatu score '])
    t1_labels = t1_labels.drop(t1_columns_to_drop, axis=1)
    #drop rows that have cats with no annotations

    #unique cat nums with their indices
    idx1 = np.unique(df1.index, return_index=True)
    t1_new_labels = return_sorted_labels(idx1,t1_labels)

    # 1 hr after surgery(t2):
    t2_labels = pd.read_excel(excel_file, sheet_name="1 hr post surg - botucatu score")
    t2_columns_to_drop = np.setdiff1d(t2_labels.columns, ['Cat', '1hr post surgery - botucatu score '])
    t2_labels = t2_labels.drop(t2_columns_to_drop, axis=1)
    # unique cat nums with their indices
    idx2 = np.unique(df2.index, return_index=True)
    t2_new_labels = return_sorted_labels(idx2, t2_labels)


    #find out max length of data points x,y per each cat. This to understand the best model dimensions







