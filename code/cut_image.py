import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import pandas as pd
import matplotlib.image as mpimg

class Crop_Image():
    def __init__(self, image_path,  an_path):
        self.image_path = image_path
        self.landmarks = pd.read_csv(an_path, header=None, sep='\t').to_numpy()
        return

    def find_vec_only_zeros(self, matrix, n):
        row_zeros = np.zeros((n))
        zero_ind = []
        for i, arr in enumerate(matrix):
            if (arr == row_zeros).all():
                zero_ind.append(i)
        return zero_ind

    def crop_im(self):
        #find points of rectangle that surrounds the face in image
        top = int(np.max(self.landmarks[:,1]))
        bottom = int(np.min(self.landmarks[:,1]))
        right = int(np.min(self.landmarks[:,0]))
        left = int(np.max(self.landmarks[:,0]))
        return top,bottom, right, left

    def draw_cropped_polygon(self):
        # create annotations graph on image

        plt.scatter(self.landmarks[:, 0], self.landmarks[:, 1], cmap='gray')
        plt.savefig("landmarks.png", cmap='gray')
        img = mpimg.imread(self.image_path)
        imgplot = plt.imshow(img)
        # save annotations image in file
        plt.savefig('ex_pic.png')
        plt.close()
        landmarks_im_arr = cv2.imread("ex_pic.png", cv2.IMREAD_GRAYSCALE)
        a, b, c, d = self.crop_im()
        cropped = landmarks_im_arr[b:a, c:d]
        img = Image.fromarray(cropped)
        img.save('polygon_ex.png')
        return

if __name__=="__main__":
    img_path = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren\Cat_pain_data_for_AI_collaboration" \
           r"\pain_no_pain_data_clinical_population\video_data\Annotated_images_sorted_by_condition" \
           r"\before_surgery_no_pain\cat_1_video_4.2.png"
    an_path =  r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren\Cat_pain_data_for_AI_collaboration" \
           r"\pain_no_pain_data_clinical_population\video_data\Annotated_images_sorted_by_condition" \
           r"\before_surgery_no_pain\cat_1_video_4.2.txt"

  #  image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
   # landmarks = pd.read_csv(an_path, header=None, sep='\t').to_numpy()

    #find and display outlines in image
    crop_landmarks = Crop_Image(img_path, an_path)
    crop_landmarks.draw_cropped_polygon()





