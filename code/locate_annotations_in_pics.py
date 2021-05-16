import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

'''This code is for presenting an example of an image with it's annotations. To see the annotations positioning on the image'''

file_path = r"C:\Users\ravit\PycharmProject\cats\Laurens_dataset\Videos From Lauren" \
            r"\Cat pain data for AI collaboration\pain_no pain data - clinical population\video data"
img_dir= "\GATA 1\cat 1 video 1"
img_name = "\cat_1_video_1.3.png"

img_path = file_path + img_dir + img_name

anotations = "\cat_1_video_1.3.txt"
anot_path = file_path + img_dir + anotations
ant = pd.read_csv(anot_path, header=None, sep='\t').to_numpy()

img = mpimg.imread(img_path)
imgplot = plt.imshow(img)

plt.show()
plt.scatter(ant[:,0], ant[:,1])




