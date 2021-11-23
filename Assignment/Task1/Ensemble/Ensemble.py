import imageio
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import csv
import cv2
import pandas as pd

from Assignment.Package import data_processing as dt

img_folder=r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/image_small'
label_file = r'/Assignment/dataset/label_small.csv'


########################################## DATA PROCESSING ######################################################
#Get images (inputs) array
images_array, class_name = dt.load_images(img_folder)
images_array = np.array(images_array)

print("\nDataset shape: {}".format(images_array.shape))
a,b,c,d = images_array.shape
print("\nImage Size: {}x{}x{}".format(b,c,d))
print("\nNumber of Images: {}".format(a))

#Get labels (outputs) array
labels = dt.load_labels(label_file)
#print(labels)
print("\nNumber of Labels: {}".format(len(labels)))

#Array to  Vectors
images_vectors = dt.image_array_to_vector(images_array)
print("\nVector Size: {}".format(len(images_vectors[0])))


#Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(images_vectors,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}'.format(round(((len(Y_train)*1.0)/len(images_vectors)),3),round((len(Y_test)*1.0)/len(labels),3)))

########################################## BAGGING CLASSIFIER ######################################################

# #1. Estimators Tunning
# BAG_accuracies_df = pd.DataFrame(list(range(1,31)), columns=["k"])
# accuracies = []
# estimators_range = [8, 12]
# for i in range(estimators_range[0],estimators_range[1]):
#     Y_pred = dt.Bagging_Classifier(X_train, Y_train, X_test,i)
#     accuracies.append(round(metrics.accuracy_score(Y_test,Y_pred),3)*100)
#
# BAG_accuracies_df['accuracies']=accuracies
# print(BAG_accuracies_df)

# #2. Estimators Visualisation
# fig, ax = plt.subplots()
# ax.scatter(BAG_accuracies_df['k'], BAG_accuracies_df['accuracies'])
# ax.set(title = 'Accuracy against number of estimators',
#         ylabel='Accuracy',xlabel='K', ylim=[50, 100])
# plt.title('Accuracy against number of number of estimators', weight = 'bold')
# plt.show()

# 3. Fit Bagging model with KNN for K = 2 and get accuracy score
Y_pred_BAG, bag_clf = dt.Bagging_Classifier(X_train, Y_train, X_test,2)
BAG_accuracy = metrics.accuracy_score(Y_test,Y_pred_BAG)
print('\nBagging Method Accuracy on test data: {}%'.format(round(BAG_accuracy*100,2)))

# 4. Plot non-normalized confusion matrix
titles_options = [
    ("Bagging Confusion matrix, without normalization", None),
    #("Bagging Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        bag_clf,
        X_test,
        Y_test,
        display_labels=["No Tumor", "Tumor"],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
    #print(title)
    #print(disp.confusion_matrix)
plt.show()

########################################## BOOSTING CLASSIFIER ######################################################

# #1. Estimators Tunning
# BOOST_accuracies_df = pd.DataFrame(list(range(1,31)), columns=["k"])
# accuracies1 = []
# estimators_range = [8, 12]
# for i in range(estimators_range[0],estimators_range[1]):
#     Y_pred = dt.BOOST_accuracy(X_train, Y_train, X_test,i)
#     accuracies1.append(round(metrics.accuracy_score(Y_test,Y_pred),3)*100)
#
# BOOST_accuracies_df['accuracies']=accuracies1
# print(BOOST_accuracies_df)

# #2. Estimators Visualisation
# fig, ax = plt.subplots()
# ax.scatter(BOOST_accuracies_df['k'], BOOST_accuracies_df['accuracies'])
# ax.set(title = 'Accuracy against number of estimators',
#         ylabel='Accuracy',xlabel='K', ylim=[92, 100])
# plt.title('Accuracy against number of number of estimators', weight = 'bold')
# plt.show()

# 3. Fit ADABOOST model with Decision Three for K = 2 and get accuracy score
Y_pred_BOOST, boost_clf = dt.Boosting_Classifier(X_train, Y_train, X_test, 2)
BOOST_accuracy = metrics.accuracy_score(Y_test,Y_pred_BOOST)
print('\nBagging Method Accuracy on test data: {}%'.format(round(BOOST_accuracy*100,2)))

# 4. Plot non-normalized confusion matrix
titles_options = [
    ("Boosting Confusion matrix, without normalization", None),
    #("Boosting Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        boost_clf,
        X_test,
        Y_test,
        display_labels=["No Tumor", "Tumor"],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
    #print(title)
    #print(disp.confusion_matrix)
plt.show()