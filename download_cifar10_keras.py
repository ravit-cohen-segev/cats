import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.utils.np_utils import to_categorical

def saveCifarImage(array, path, file):

    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path+file+".tiff", array)

def reshape_and_print(cifar_data):
    # number of images in rows and columns
    rows = cols = np.sqrt(cifar_data.shape[0]).astype(np.int32)
    # Image hight and width. Divide by 3 because of 3 color channels
    imh = imw = np.sqrt(cifar_data.shape[1] // 3).astype(np.int32)
    # reshape to number of images X color channels X image size
    # transpose to color channels X number of images X image size
    timg = cifar_data.reshape(rows * cols, 3, imh * imh).transpose(1, 0, 2)
    # reshape to color channels X rows X cols X image hight X image with
    # swap axis to color channels X rows X image hight X cols X image with
    timg = timg.reshape(3, rows, cols, imh, imw).swapaxes(2, 3)
    # reshape to color channels X combined image hight X combined image with
    # transpose to combined image hight X combined image with X color channels
    timg = timg.reshape(3, rows * imh, cols * imw).transpose(1, 2, 0)

    plt.imshow(timg)
    plt.show()

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#indices of images that belong to 'cat' class
train_index = np.where(y_train[:,0] == 3)[0]
test_index = np.where(y_test[:,0] == 3)[0]

x_tr = [X_train[idx] for idx in train_index]
y_tr = [y_train[idx] for idx in train_index]

x_te = [X_test[idx] for idx in test_index]
y_te = [y_test[idx] for idx in test_index]

path = r"C:\Users\ravit\PycharmProject\cats\data\cifar10\cat"

l = 10
#l=x_tr

reshape_and_print(x_tr[1])

#saving train images
#for i, arr in enumerate(l):
 #   saveCifarImage(arr, path, "/train\cat"+str(i))

#saving test images
#for i, arr in enumerate(x_te):
 #   saveCifarImage(arr, path, "/test\cat"+str(i))