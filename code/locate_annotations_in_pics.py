import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os

def save_annotations_in_image(path, file, image_file, save_dir):

    anot_path = os.path.join(path, file)
    img_path = os.path.join(path,image_file_name)
    ant = pd.read_csv(anot_path, header=None, sep='\t').to_numpy()


    #display image
    img = mpimg.imread(img_path)
    imgplot = plt.imshow(img)

    # create annotations graph on image
    plt.scatter(ant[:, 0], ant[:, 1])
    plt.show()
    #change dir to save in the right place
    #save annotations image in file
    plt.savefig(save_dir+'_'+image_file_name)
    #close graph
    plt.close()
    return

if __name__=="__main__":

    file_path = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren" \
                r"\Cat_pain_data_for_AI_collaboration\pain_no_pain_data_clinical_population" \
                r"\video_data\Annotated_images_sorted_by_condition"

    img_dir = ["before_surgery_no_pain","1_hour_after_surgery_worst_pain"]
  #  img_name = "_cat_30_video_1.1.png"

  #  img_path = os.path.join(file_path,img_dir,img_name)

   # anotations = "_cat_30_video_1.1"
   # ext = '.txt'

   # dir_save = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos_From_Lauren" \
    #           r"\Cat_pain_data_for_AI_collaboration\pain_no_pain_data_-_clinical_population\video_data" \
     #          r"\anot_images_for_DL"


   # path = os.path.join(file_path, img_dir)
   # save_annotations_in_image(path, anotations, ext, dir_save)

    for dir in img_dir:
        path = os.path.join(file_path, dir)
        dir_save = os.path.join(file_path, "for_DL", dir)
        for root, dirs, files in os.walk(path, topdown=False):
            for file in files:
                if '.txt' in file:
                    # get image file name
                    image_file_name = file.replace('.txt', '.png')
                    if image_file_name not in files:
                        continue
                    save_annotations_in_image(path, file, image_file_name, dir_save)

    # to graphically show an example of annotations in an image

    #    img = mpimg.imread(img_path)
    #   imgplot = plt.imshow(img)

    #  plt.show()
    # plt.scatter(ant[:, 0], ant[:, 1])





