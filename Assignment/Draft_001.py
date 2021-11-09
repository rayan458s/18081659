import imageio
import mpimg as mpimg
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import skimage
from PIL import Image
import random


IMG_WIDTH=200
IMG_HEIGHT=200
img_folder=r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/image_test'

def create_dataset_PIL(img_folder):
    img_data_array=[]
    class_name=[]
    for file in os.listdir(img_folder):     #for all the files in dataset/image
        print('Loading {}'.format(file))
        image_path= os.path.join(img_folder, file)      #join the path to the image filename
        image= np.array(Image.open(image_path))             #open and convert to numpy array
        image= np.resize(image,(IMG_HEIGHT,IMG_WIDTH,3))        #rescale
        image = image.astype('float32')                         #converto to float
        image /= 255
        img_data_array.append(image)                    #final list with all the image arrays
        class_name.append(file)                             #image names
    return img_data_array , class_name

PIL_img_data, class_name = create_dataset_PIL(img_folder)
#print(class_name)
print("\nTotal number of images: {}".format(len(PIL_img_data)))
print("\nImage size: {}x{}x{}\n".format(len(PIL_img_data[0]), len(PIL_img_data[0][0]),  len(PIL_img_data[0][0][0]) ))

#print(PIL_img_data[0])


plt.figure(figsize=(20,20))

for i in range(5):
    file = random.choice(os.listdir(img_folder))
    image_path= os.path.join(img_folder, file)
    img = np.array(imageio.imread(image_path))
    ax=plt.subplot(1,5,i+1)
    ax.title.set_text(file)
    plt.imshow(img)

print(img)
plt.show()


print("\nImage size: {}x{}x{}\n".format(len(img), len(img[0]),  len(img[0][0]) ))



