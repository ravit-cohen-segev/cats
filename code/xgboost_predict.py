import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
from landmark_pred_pain_logistic_random_forest_svm import *
import xgboost as xgb
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    dir_path = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren" \
               r"\Cat_pain_data_for_AI_collaboration\pain_no_pain_data_clinical_population" \
               r"\video_data\Annotated_images_sorted_by_condition"

    dir_list = ['1_hour_after_surgery_worst_pain', 'before_surgery_no_pain']

    np.random.seed(42)

    DATA_build = DATA_landmarks(dir_path)
    landmarks_euc_array, labels = DATA_build.create_array_landmarks(dir_list, cat_nums=True)

    # save euclidean array matrix to file
    df_landmarks = pd.DataFrame(landmarks_euc_array)
    df_labels = pd.DataFrame(labels)
    df_landmarks.to_csv("landmark_euclidean_arr.csv")
    df_labels.to_csv("labels.csv")

    # save labels as int array
    y = labels['label'].values.astype('int')

    X_train, X_test, y_train, y_test = train_test_split(landmarks_euc_array, y, test_size=0.25, random_state=42)

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    cross_val, train_accuracy, test_accuracy = predict_accuracy(model, X_train, y_train, X_test, y_test)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions with sklearn
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    #sccuracy for train is 100% and 74% for test
