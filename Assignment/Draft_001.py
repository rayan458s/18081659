import imageio
import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import skimage
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from matplotlib import pyplot
from matplotlib.image import imread
import csv
import cv2


IMG_WIDTH=200
IMG_HEIGHT=200
img_folder=r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/image'
label_file =r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/label.csv'


def plot_4_images():
    plt.figure(figsize=(20,20))
    for i in range(4):
        file = random.choice(os.listdir(img_folder))
        image_path= os.path.join(img_folder, file)
        img = np.array(imageio.imread(image_path))
        ax=plt.subplot(2,2,i+1)
        ax.title.set_text(file)
        plt.imshow(img)
    plt.show()


def load_images(img_folder):
    images_array=[]
    class_name=[]
    for file in os.listdir(img_folder):     #for all the files in dataset/image
        #print('Loading {}'.format(file))
        image_path = os.path.join(img_folder, file)      #join the path to the image filename
        image = np.array(imageio.imread(image_path))             #open and convert to numpy array
        #image= np.resize(image,(IMG_HEIGHT,IMG_WIDTH,3))        #rescale
        #image = image.astype('float32')                         #converto to float
        #image /= 255
        images_array.append(image)                    #final list with all the image arrays
        class_name.append(file)                             #image names
    return images_array , class_name


def load_labels(label_file_path):

    open_file = open(label_file_path)
    read_file = csv.reader(open_file, delimiter=',')
    labels = []
    for row in read_file:
        if row[1] == 'no_tumor':
           labels.append(0)
        else:
            labels.append(1)
    labels.pop(0)
    labels = np.array(labels)
    return labels

def image_to_feature_vector(images_array, size=(512, 512)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
    image_vectors = []
    for i in range(len(images_array)):
        image = images_array[i]
        image_vector = cv2.resize(image, size).flatten()
        image_vectors.append(image_vector)
    image_vectors = np.array(image_vectors)

    return image_vectors

def KNN_Classifier(X_train, y_train, X_test,k):

    #Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train) # Fit KNN model


    Y_pred = neigh.predict(X_test)
    return Y_pred

#Get images (inputs) array
images_array, class_name = load_images(img_folder)
images_array = np.array(images_array)

print("\nDataset shape: {}".format(images_array.shape))
a,b,c,d = images_array.shape
print("\nNumber of Images: {}".format(a))
print("\nImage Size: {}x{}x{}".format(b,c,d))

#Get labels (outputs) array
labels = load_labels(label_file)
#print(labels)
print("\nNumber of Labels: {}".format(len(labels)))

#Array to Feature Vectors
image_vectors = image_to_feature_vector(images_array)
print("\nNumber of Images: {}".format(len(image_vectors)))
print("\nFeature Vector Size: {}".format(len(image_vectors[0])))


#Split train an test dataset
X_train,X_test,y_train,y_test=train_test_split(image_vectors,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}'.format(round(((len(y_train)*1.0)/len(image_vectors)),3),round((len(y_test)*1.0)/len(labels),3)))

Y_pred = KNN_Classifier(X_train, y_train, X_test,10)
print(round(metrics.accuracy_score(y_test,Y_pred),3)*100)
print(y_test)
print(Y_pred)

count_ytest = np.ndarray.tolist(y_test)
count_ypred = np.ndarray.tolist(Y_pred)



print('\nNumber of 0 in y_test; {}'.format(count_ytest.count(0)))
print('\nNumber of 0 in y_pred; {}'.format(count_ypred.count(0)))
print('\nNumber of 1 in y_test; {}'.format(count_ytest.count(1)))
print('\nNumber of 1 in y_pred; {}'.format(count_ypred.count(1)))









