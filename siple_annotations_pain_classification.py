import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

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

    def create_array_landmarks(self, dir_list, cat_nums=True, feature_num=48):
        #cat_nums is used as id for the clinical
        # population because the files are arranged differently
        #calc vec length
        #count = 0
        #for i in range(1,feature_num):
         #   count += i
          # count is 1128
        columns = ['cat', 'label']
        labels_df = pd.DataFrame(columns = columns)
        euc_arr = np.empty((0,1128))
        idx_count = 0
        #if not cat_nums, that is not from the clinical population. Use a counting variable for cat id
        cat_count = 0
        for i, dir in enumerate(dir_list):
            full_path = os.path.join(self.path, dir)
        #    all_files = os.listdir(full_path)

            for root,d,files in os.walk(full_path):
                for file in files:
                    #check if file is with '.txt' extension. if not than continue to the next file
                    if '.txt' not in file:
                        continue
                    #if dir is empty
                    if d==[]:
                        anot_path = os.path.join(root,file)
                    #if dir is not empty
                    else:
                        anot_path = os.path.join(root,d,file)
                    #read annotations
                    df_annot = pd.read_csv(anot_path, header=None, sep='\t').T
                    #add row with cat num and pain clasification
                    # add label, if t1 than no pain (0) if t2 than pain (1)
                    if cat_nums:
                        labels_df.loc[idx_count] = [self.check_cat_name_in_file(file), i]
                    else:
                        labels_df.loc[idx_count] = [cat_count, i]
                        cat_count += 1
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

class Logit():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        return

    def create_logistic_model(self):
        self.model = LogisticRegression(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def predict_accuracy(self, model):
        self.create_logistic_model()
        # cross validation
        cross_val = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1_macro')
        train_accuracy = model.score(self.X_train, self.y_train)
        test_accuracy = model.score(self.X_test, self.y_test)
        return cross_val, train_accuracy, test_accuracy


if __name__=="__main__":
    dir_path = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren" \
               r"\Cat_pain_data_for_AI_collaboration\pain_no_pain_data_clinical_population" \
               r"\video_data\Annotated_images_sorted_by_condition"

    dir_list = ['1_hour_after_surgery_worst_pain', 'before_surgery_no_pain']

    np.random.seed(0)

    DATA_build = DATA_landmarks(dir_path)
    landmarks_euc_array, labels = DATA_build.create_array_landmarks(dir_list, cat_nums=True)

    #save labels as int array
    y = labels['label'].values.astype('int')

    X_train, X_test, y_train, y_test = train_test_split(landmarks_euc_array, y, test_size = 0.25, random_state = 42)

    #create class instance for logistic
    logistic = Logit(X_train, y_train, X_test, y_test)
    logistic_model = logistic.create_logistic_model()
    cross_val, train_acc_no_pca, test_acc_no_pca = logistic.predict_accuracy(logistic_model)

    predict_train_no_pca = logistic_model.predict(X_train)
    predict_test_no_pca = logistic_model.predict(X_test)


    #create logistic model after performing PCA
#    pca = PCA(n_components=150)
 #   new_data = pca.fit_transform(landmarks_euc_array)
    #split new data
  #  X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(new_data, y, test_size=0.25, random_state=42)
    #create another instance for logisitc_pca
   # logistic_pca = Logit(X_train_pca, y_train_pca, X_test_pca, y_test_pca)
   # logistic_pca_model = logistic_pca.create_logistic_model()

   # cross_val_pca, train_acc_pca, test_acc_pca = logistic_pca.predict_accuracy(logistic_pca_model)

   # predict_train_pca = logistic_pca_model.predict(X_train_pca)
   # predict_test_pca = logistic_pca_model.predict(X_test_pca)

    ####################################################################################################################
    #Add cats with and without pain from the diverse file to analysis
    ####################################################################################################################

    diverse_path = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren" \
                   r"\Cat_pain_data_for_AI_collaboration\pain_no_pain_data_diverse_population_"

    #changed the order of directories to have 'no_pain' at postion 0. This is important for labeling
    dir_list = ['no_pain', 'pain']

    div_DATA = DATA_landmarks(diverse_path)
    landmarks_euc_array_div, labels_div = div_DATA.create_array_landmarks(dir_list, cat_nums=False)

    #combine diverse data with clinical data
    combined_data = np.vstack((landmarks_euc_array, landmarks_euc_array_div))
    combined_labels = np.append(labels['label'].values.astype('int'), labels_div['label'].values.astype('int'))

    #split to train and test
    X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(combined_data, combined_labels,
                                                                        test_size=0.25, random_state=42)
    #train logistic
    logistic_all = Logit(X_all_train, y_all_train, X_all_test, y_all_test)
    logistic_all_model = logistic_all.create_logistic_model()

    cross_val_all, train_acc_all, test_acc_all = logistic_all.predict_accuracy(logistic_all_model)

    predict_train_all = logistic_all_model.predict(X_all_train)
    predict_test_all = logistic_all_model.predict(X_all_test)




