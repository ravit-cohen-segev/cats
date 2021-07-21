# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 23:12:01 2021

@author: ravit
"""

#taken from git repository https://github.com/koshian2/cats-face-landmarks/blob/master/train.py

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, History
import tensorflow.keras.backend as K
from keras.objectives import mean_squared_error
from PIL import Image
import pickle, glob, random, zipfile
from create_dataset_functions import *

def enumerate_layers():
    resnet = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    resnet.summary()
    for i, layer in enumerate(resnet.layers):
        print(i, layer.name)

def create_resnet():
    resnet = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    for i in range(48):
     
        resnet.layers[i].trainable=False 

    x = GlobalAveragePooling2D()(resnet.output)
 
    x = Dense(18, activation="sigmoid")(x)
    model = Model(resnet.inputs, x)
    return model

class CatGenerator:
    def __init__(self):
        pass

    def flow_from_directory(self, batch_size, train=True, shuffle=True, use_data_augmentation=True):
        dir_path_clinical = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren\Cat_pain_data_for_AI_collaboration\pain_no_pain_data_clinical_population\video_data\Annotated_images_sorted_by_condition"
        dir_path_div_pain = r"C:/Users/ravit/PycharmProject/cats/Laurens_dataset/Videos_From_Lauren/Cat_pain_data_for_AI_collaboration/pain_no_pain_data_diverse_population_/pain"
        dir_path_div_no_pain = r"C:/Users/ravit/PycharmProject/cats/Laurens_dataset/Videos_From_Lauren/Cat_pain_data_for_AI_collaboration/pain_no_pain_data_diverse_population_/no_pain"

        # define image shape for reshaping all of the images in the dataset
        # the image shape is defined by the original code
        image_shape = (224, 224)

        create_data = create_dataset_dict(dir_path_clinical, image_shape)
        x_data, y_data = create_data.create_dict()

        create_data_div_pain = create_dataset_dict(dir_path_div_pain, image_shape)
        x_data_div_pain, y_data_div_pain = create_data_div_pain.create_dict()

        create_data_div_no_pain = create_dataset_dict(dir_path_div_no_pain, image_shape)
        x_data_div_no_pain, y_data_div_no_pain = create_data_div_no_pain.create_dict()

        x_data.extend(x_data_div_pain)
        x_data.extend(x_data_div_no_pain)

        y_data.extend(y_data_div_pain)
        y_data.extend(y_data_div_no_pain)
        '''
        if len(X_cache) == batch_size:
            X_batch = np.asarray(X_cache, dtype=np.float32) / 255.0
            y_batch = np.asarray(y_cache, dtype=np.float32)
            X_cache, y_cache = [], []
            yield X_batch, y_batch'''
        return x_data, y_data

def loss_function_simple(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def loss_function_with_distance(y_true, y_pred):
    point_mse = mean_squared_error(y_true, y_pred)
    distance_mse = mean_squared_error(y_true[:, 2:18]-y_true[:, 0:16], y_pred[:, 2:18]-y_pred[:, 0:16])
    return point_mse + distance_mse

def loss_function_with_multiple_distance(y_true, y_pred):
    error = mean_squared_error(y_true, y_pred)
    for i in range(8):
        error += mean_squared_error(y_true[:, ((i+1)*2):18]-y_true[:, 0:(16-i*2)], y_pred[:, ((i+1)*2):18]-y_pred[:, 0:(16-i*2)])
    return error

def sarrus_formula(p1, p2, p3):
    a = p2 - p1
    b = p3 - p1
    return K.abs(a[:,0]*b[:,1] - a[:,1]*b[:,0]) / 2.0

from itertools import combinations

def loss_function_multiple_distance_and_triangle(y_true, y_pred):
    error = mean_squared_error(y_true, y_pred)
    for i in range(8):
        error += mean_squared_error(y_true[:, ((i+1)*2):18]-y_true[:, 0:(16-i*2)], y_pred[:, ((i+1)*2):18]-y_pred[:, 0:(16-i*2)])

    for comb in combinations(range(9), 3):
        s_true = sarrus_formula(
            y_true[:, (comb[0]*2):(comb[0]*2+2)],
            y_true[:, (comb[1]*2):(comb[1]*2+2)],
            y_true[:, (comb[2]*2):(comb[2]*2+2)]
        )
        s_pred = sarrus_formula(
            y_pred[:, (comb[0]*2):(comb[0]*2+2)],
            y_pred[:, (comb[1]*2):(comb[1]*2+2)],
            y_pred[:, (comb[2]*2):(comb[2]*2+2)]
        )
        error += K.abs(s_true - s_pred)
    return error

def calc_area_loss(ear_true, ear_pred):
    left_x = K.expand_dims(K.min(ear_true[:, ::2], axis=-1))
    left_y = K.expand_dims(K.min(ear_true[:, 1::2], axis=-1))
    right_x = K.expand_dims(K.max(ear_true[:, ::2], axis=-1))
    right_y = K.expand_dims(K.max(ear_true[:, 1::2], axis=-1))

    pred_x = ear_pred[:, ::2]
    pred_y = ear_pred[:, 1::2]
 
    penalty_x = K.maximum(left_x - pred_x, 0.0) + K.maximum(pred_x - right_x, 0.0)
    penalty_y = K.maximum(left_y - pred_y, 0.0) + K.maximum(pred_y - right_y, 0.0)
    return K.mean(penalty_x + penalty_y, axis=-1)

def loss_function_multiple_distance_and_area(y_true, y_pred):

    error = mean_squared_error(y_true, y_pred)
 
    for i in range(8):
        error += mean_squared_error(y_true[:, ((i+1)*2):18]-y_true[:, 0:(16-i*2)], y_pred[:, ((i+1)*2):18]-y_pred[:, 0:(16-i*2)])
   
    left_ear_true, left_ear_pred = y_true[:, 6:12], y_pred[:, 6:12]
    right_ear_true, right_ear_pred = y_true[:, 12:18], y_pred[:, 12:18]
    error += calc_area_loss(left_ear_true, left_ear_pred)
    error += calc_area_loss(right_ear_true, right_ear_pred)
    return error

class CatsCallback(Callback):
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.min_val_loss = np.inf

    def on_train_begin(self, logs):
        self.reset()

    def on_epoch_end(self, epoch, logs):
        if logs["val_loss"] < self.min_val_loss:
                self.model.save_weights("./cats_weights.hdf5", save_format="h5")
                self.min_val_loss = logs["val_loss"]
                print("Weights saved.", self.min_val_loss)

               
def train(batch_size, load_existing_weights):
    model = create_resnet()
    gen = CatGenerator()

    if load_existing_weights:
        model.load_weights("weights.hdf5")

    model.compile(tf.compat.v1.train.MomentumOptimizer(1e-3, 0.9), loss=loss_function_multiple_distance_and_area, metrics=[loss_function_simple])

    cb = CatsCallback(model)
    history = History()

    model.fit_generator(gen.flow_from_directory(batch_size, True), validation_data=gen.flow_from_directory(batch_size, False), callbacks=[cb, history], epochs=200)

    with open("history.dat", "wb") as fp:
        pickle.dump(history.history, fp)

    with zipfile.ZipFile("cats_result.zip", "w") as zip:
        zip.write("history.dat")
        zip.write("cats_weights.hdf5")


if __name__ == "__main__":
    # check GPU available
    print('Version of tensorflow:\n', tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))

    if tf.test.is_gpu_available():
        device_name = tf.test.gpu_device_name()
    else:
        device_name = '/CPU:0'
    print(device_name)

    train(173, False)


