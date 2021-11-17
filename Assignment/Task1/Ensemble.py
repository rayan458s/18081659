import imageio
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import csv
import cv2
import pandas as pd


IMG_WIDTH=200
IMG_HEIGHT=200
img_folder=r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/image_small'
label_file = r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/label_small.csv'


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


def Bagging_Classifier(X_train, y_train, X_test,k):
    bagmodel=BaggingClassifier(n_estimators=k,max_samples=0.5, max_features=4,random_state=1)  #Create KNN object with a K coefficient
    bagmodel.fit(X_train, y_train)      #Fit KNN model
    Y_pred_BAG = bagmodel.predict(X_test)
    return Y_pred_BAG


def Boosting_Classifier(X_train, Y_train, X_test,k):
    boostmodel=AdaBoostClassifier(n_estimators=k)       # AdaBoost takes Decision Tree as its base-estimator model by default.
    boostmodel.fit(X_train,Y_train,sample_weight=None)  # Fit KNN model
    Y_pred_BOOST = boostmodel.predict(X_test)
    return Y_pred_BOOST


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
X_train,X_test,Y_train,Y_test=train_test_split(image_vectors,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}'.format(round(((len(Y_train)*1.0)/len(image_vectors)),3),round((len(Y_test)*1.0)/len(labels),3)))

########################################## BAGGING CLASSIFIER ######################################################
Y_pred_BAG = Bagging_Classifier(X_train, Y_train, X_test,2)
BAG_accuracy = metrics.accuracy_score(Y_test,Y_pred_BAG)
print('\nBagging Method Accuracy on test data: {}%'.format(round(BAG_accuracy*100,2)))

# #Estimators Tunning
# BAG_accuracies_df = pd.DataFrame(list(range(1,31)), columns=["k"])
# accuracies = []
# estimators_range = [8, 12]
# for i in range(estimators_range[0],estimators_range[1]):
#     Y_pred = Bagging_Classifier(X_train, Y_train, X_test,i)
#     accuracies.append(round(metrics.accuracy_score(Y_test,Y_pred),3)*100)
#
# BAG_accuracies_df['accuracies']=accuracies
# print(BAG_accuracies_df)
#
#Estimators Visualisation
# fig, ax = plt.subplots()
# ax.scatter(BAG_accuracies_df['k'], BAG_accuracies_df['accuracies'])
# ax.set(title = 'Accuracy against number of estimators',
#         ylabel='Accuracy',xlabel='K', ylim=[50, 100])
# plt.title('Accuracy against number of number of estimators', weight = 'bold')
# plt.show()

########################################## BOOSTING CLASSIFIER ######################################################
Y_pred_BOOST = Boosting_Classifier(X_train, Y_train, X_test, 2)
BOOST_accuracy=metrics.accuracy_score(Y_test,Y_pred_BOOST)
print('\nBoosting Method Accuracy on test data: {}%'.format(round(BOOST_accuracy*100,2)))

# #Estimators Tunning
# BOOST_accuracies_df = pd.DataFrame(list(range(1,31)), columns=["k"])
# accuracies1 = []
# estimators_range = [8, 12]
# for i in range(estimators_range[0],estimators_range[1]):
#     Y_pred = BOOST_accuracy(X_train, Y_train, X_test,i)
#     accuracies1.append(round(metrics.accuracy_score(Y_test,Y_pred),3)*100)
#
# BOOST_accuracies_df['accuracies']=accuracies1
# print(BOOST_accuracies_df)
#
# #Estimators Visualisation
# fig, ax = plt.subplots()
# ax.scatter(BOOST_accuracies_df['k'], BOOST_accuracies_df['accuracies'])
# ax.set(title = 'Accuracy against number of estimators',
#         ylabel='Accuracy',xlabel='K', ylim=[92, 100])
# plt.title('Accuracy against number of number of estimators', weight = 'bold')
# plt.show()
