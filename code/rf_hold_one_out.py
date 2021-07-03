import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

class multi_rf_landmarks():
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

    def normalized_matrix(self, arr):
        normalized = np.empty((0, arr.shape[1]))
        for i, row in enumerate(arr):
            max = np.max(row)
            normalized = np.vstack((normalized, row / max))
        return normalized
    
    def comp_roc_curve(self, y_tr, y_te):
        tpr = 0
        fpr = 0
        for i, val in enumerate(y_tr):
            if y_te == 1:
                if val == 1:
                    tpr += 1
                else:
                    fpr += 1
        return tpr, fpr
        
    def plot_roc(self, tpr_d, fpr_d):
        plt.figure()
        lw = 2
        plt.plot(fpr_d, tpr_d, color='darkorange',
        lw=lw, label='ROC curve (area = %0.2f)')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()
        
       

    def create_split(self, i_test):
        columns = ['cat', 'label']
        #create train and test empty dfs
        test_euc_arr = np.empty((0,1128))
        test_labels = pd.DataFrame(columns=columns)
        train_euc_arr = np.empty((0,1128))
        train_labels = pd.DataFrame(columns=columns)


        idx_train_count = 0
        idx_test_count = 0
        #add data to test and train datasets
        for i, dir in enumerate(dir_list):
            full_path = os.path.join(self.path, dir)
            #    all_files = os.listdir(full_path)

            for root, d, files in os.walk(full_path):
                for file in files:
                    # check if file is with '.txt' extension. if not than continue to the next file
                    if '.txt' not in file:
                        continue
                    # if dir is empty
                    if d == []:
                        anot_path = os.path.join(root, file)
                    # if dir is not empty
                    else:
                        anot_path = os.path.join(root, d, file)
                    # read annotations
                    df_annot = pd.read_csv(anot_path, header=None, sep='\t').T
                    # convert to numpy
                    arr_from_df = df_annot.to_numpy()
                    euc_vec = self.calc_euc_vec(arr_from_df)
                    # add row with cat num and pain clasification
                    # add label, if t1 than no pain (0) if t2 than pain (1)
                    max = np.max(euc_vec)
                    if '_'+str(i_test)+'_' in file:
                        test_labels.loc[idx_test_count] = [i_test, i]
                        idx_test_count += 1

                        test_euc_arr = np.vstack((test_euc_arr, euc_vec))


                    else:
                        train_labels.loc[idx_train_count] = [self.check_cat_name_in_file(file), i]
                        idx_train_count += 1
                        train_euc_arr = np.vstack((train_euc_arr, euc_vec))
                    # normalize test and train data by max value in each row

                    normalized_train_euc = self.normalized_matrix(train_euc_arr)
                    normalized_test_euc = self.normalized_matrix(test_euc_arr)
        return normalized_train_euc, train_labels, normalized_test_euc, test_labels


    def run_multiple_rf(self):
        res = pd.DataFrame(columns=["train_acc", "test_acc"])
        #if not cat_nums, that is not from the clinical population. Use a counting variable for cat id
        forest_model = RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=2)
        tpr_d = {}
        fpr_d = {}
        count = 0
        #
        
        #iterate over cats numbers
        for i in range(1,31):
            X_train, y_train, X_test, y_test = self.create_split(i)
            if len(X_train) == 0 or len(X_test) == 0:
                print('empty array:', i)
                continue
            forest_model.fit(X_train, y_train['label'].values.astype('float'))
           # rf_train_acc, rf_test_acc = predict_accuracy(forest_model, X_train, y_train['label'].values.astype('float')
            #                                             , X_test, y_test['label'].values.astype('float'))
                                
          #  print("train_acc: {}, test_acc: {} for {} cat as test".format(rf_train_acc, rf_test_acc, str(i)))
          #  res.loc[count] = [rf_train_acc, rf_test_acc]
            tpr_d[count], fpr_d[count] = self.comp_roc_curve(y_train, y_test)
            
            count += 1
        
        self.plot_roc(tpr_d, fpr_d)
        return 

def predict_accuracy(model, X_train, y_train, X_test, y_test):
    # cross validation
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    return train_accuracy, test_accuracy

if __name__=="__main__":
    dir_path = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren" \
               r"\Cat_pain_data_for_AI_collaboration\pain_no_pain_data_clinical_population" \
               r"\video_data\Annotated_images_sorted_by_condition"


    dir_list = ['1_hour_after_surgery_worst_pain', 'before_surgery_no_pain']
    np.random.seed(42)

    multi_rf = multi_rf_landmarks(dir_path)
    multi_rf.run_multiple_rf()
  #  res_file = multi_rf.run_multiple_rf()
 #   res_file.to_excel("loocv_rf.xlsx")
