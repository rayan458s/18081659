import imageio
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, chi2
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
img_folder=r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/image'
label_file = r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/label.csv'


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

def image_array_to_vector(images_array, size=(512, 512)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
    image_vectors = []
    for i in range(len(images_array)):
        image = images_array[i]
        image_vector = cv2.resize(image, size).flatten()
        image_vectors.append(image_vector)
    image_vectors = np.array(image_vectors)
    return image_vectors

def PCAPredict(X, k):
    '''
    Inputs
        X: dataset;
        k: number of Components.

    Return
        SValue: The singular values corresponding to each of the selected components.
        Variance: The amount of variance explained by each of the selected components.
                It will provide you with the amount of information or variance each principal component holds after projecting the data to a lower dimensional subspace.
        Vcomp: The estimated number of components.
    '''
    pca = PCA(n_components=k)   # the built-in function for PCA, where n_clusters is the number of clusters.
    pca.fit(X)      # fit the algorithm with dataset

    Variance = pca.explained_variance_ratio_
    SValue = pca.singular_values_
    Vcomp = pca.components_
    return SValue, Variance, Vcomp


def KNN_Classifier(X_train, Y_train, X_test,k):
    neigh = KNeighborsClassifier(n_neighbors=k)     #Create KNN object with a K coefficient
    neigh.fit(X_train, Y_train) # Fit KNN model
    Y_pred_KNN = neigh.predict(X_test)
    return Y_pred_KNN


########################################## DATA PROCESSING ######################################################
#Get images (inputs) array
images_array, class_name = load_images(img_folder)
images_array = np.array(images_array)

print("\nDataset shape: {}".format(images_array.shape))
a,b,c,d = images_array.shape
print("\nImage Size: {}x{}x{}".format(b,c,d))
print("\nNumber of Images: {}".format(a))

#Get labels (outputs) array
labels = load_labels(label_file)
#print(labels)
print("\nNumber of Labels: {}".format(len(labels)))

#Array to Feature Vectors
images_vectors = image_array_to_vector(images_array)
print("\nVector Size: {}".format(len(images_vectors[0])))

#Dimension Reduction using PCA (feature extraction)
k1 = 10
SingularValue, Variance, Vcomponent = PCAPredict(images_vectors,k1)
#print("\nSingular Value: {}".format(SingularValue))
#print("\nVariance: {}".format(Variance))
#print("\nVcomponent: {}".format(Vcomponent))

images_features = []
single_image_feature = []
for image_vector in images_vectors:
    for component in Vcomponent:
        single_image_feature.append(abs(np.dot(image_vector,component)))
    images_features.append(single_image_feature)
    single_image_feature = []

print("\nNumber of features: {}".format(k1))
print("\nShape: {}".format(np.array(images_features).shape))

#Feature Selection
best_images_features = SelectPercentile(chi2, percentile=10).fit_transform(images_features, labels)
print(best_images_features.shape)

#Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(best_images_features,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}'.format(round(((len(Y_train)*1.0)/len(best_images_features)),3),round((len(Y_test)*1.0)/len(labels),3)))

########################################## KNN CLASSIFIER ######################################################

# #Fit KNN model for different values of k
# estimators = [1,20]
# accuracies_df = pd.DataFrame(list(range(1,estimators[1])), columns=["k"])
# accuracies = []
# for i in range(estimators[0],estimators[1]):
#     Y_pred_KNN = KNN_Classifier(X_train, Y_train, X_test,i)
#     accuracies.append(round(accuracy_score(Y_test,Y_pred_KNN),3)*100)
#     print('Accuracy for k={} computed'.format(i))
#
# accuracies_df['accuracies']=accuracies
# print('KNN Accuracy Score on Test data:\n',accuracies_df)
# print('The value of K should be below {}'.format(round(np.sqrt(len(X_train)))))
#
# #Plot accuracy vs k
# fig, ax = plt.subplots()
# ax.scatter(accuracies_df['k'], accuracies_df['accuracies'])
# ax.set(title = 'Accuracy against number of neighbors K',
#             ylabel='Accuracy (%)',xlabel='Number of Estimators K', ylim=[60, 100])
# plt.title('Accuracy against number of neighbors K', weight = 'bold')
# plt.show()

# Fit KNN model for K = 10 and get accuracy score
Y_pred = KNN_Classifier(X_train, Y_train, X_test,16)
print('KNN Accuracy Score on Test data: {}\n'.format(round(metrics.accuracy_score(Y_test,Y_pred),3)*100))


# Get information for confusion matrix
positives = 0
negatives = 0
true_positives = 0
false_negatives = 0
true_negatives = 0
false_positives = 0
for i in range(len(Y_test)):
    if Y_pred[i] == 1:
        positives += 1
        if Y_test[i] == 1:
            true_positives += 1
        else:
            false_positives += 1
    if Y_pred[i] == 0:
        negatives += 1
        if Y_test[i] == 0:
            true_negatives += 1
        else:
            false_negatives += 1

print('Predicted Positives: {}'.format(positives))
print('True positives: {}'.format(true_positives))
print('False positives: {}'.format(false_positives))

print('Predicted Negatives: {}'.format(negatives))
print('True negatives: {}'.format(true_negatives))
print('False negatives: {}\n'.format(false_negatives))

