import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import pandas as pd

class Crop_Image():
    def __init__(self, image,  landmarks):
        self.image = image
        self.landmarks = landmarks
        #self.image = cv2.resize(image, (403,658))
        return

    def find_vec_only_zeros(self, matrix, n):
        row_zeros = np.zeros((n))
        zero_ind = []
        for i, arr in enumerate(matrix):
            if (arr == row_zeros).all():
                zero_ind.append(i)
        return zero_ind

    def crop_im(self):
        r, c = self.image.shape
        #return row indices for rows with zeros exclusively
        idx_rows = self.find_vec_only_zeros(self.image,c)
        idx_cols = self.find_vec_only_zeros(self.image.T, r)
        cropped = np.delete(self.image, idx_rows, axis=0)
        cropped = np.delete(cropped, idx_cols, axis=1)
        return cropped

    def draw_cropped_polygon(self):

        im_cropped = self.crop_im()
        #draw polygon with landmarks outlines
        img = Image.fromarray(im_cropped)
        draw = ImageDraw.Draw(img)
        outlines = landmarks.flatten().tolist()
        draw.polygon(outlines)
        img.save('polygon_ex.png')
        return

if __name__=="__main__":
    img_path = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren\Cat_pain_data_for_AI_collaboration" \
           r"\pain_no_pain_data_clinical_population\video_data\Annotated_images_sorted_by_condition" \
           r"\before_surgery_no_pain\cat_1_video_4.2.png"
    an_path =  r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren\Cat_pain_data_for_AI_collaboration" \
           r"\pain_no_pain_data_clinical_population\video_data\Annotated_images_sorted_by_condition" \
           r"\before_surgery_no_pain\cat_1_video_4.2.txt"

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    landmarks = pd.read_csv(an_path, header=None, sep='\t').to_numpy()
    #find and display outlines in image
    crop_landmarks = Crop_Image(image, landmarks)
    crop_landmarks.draw_cropped_polygon()





