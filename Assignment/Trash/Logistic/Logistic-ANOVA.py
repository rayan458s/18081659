import imageio
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import csv
import cv2
import pandas as pd

from Assignment.Functions import data_processing_T1 as dt

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

#Select 10 Features using ANOVA
k_ANOVA = 10
images_features = dt.select_features_with_ANOVA(images_vectors, labels, k_ANOVA)
print("\nSelected number of features: {}".format(k_ANOVA))
print("\nFinal Input Data Shape: {}".format(np.array(images_features).shape))

#Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(images_features,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}'.format(round(((len(Y_train)*1.0)/len(images_features)),3),round((len(Y_test)*1.0)/len(labels),3)))

#Plot the features importances
forest_importances, std = dt.get_features_importance_with_RF(X_train, Y_train)
fig, ax = plt.subplots()            #define the plot object
forest_importances.plot.bar(yerr=std, ax=ax)        #plot ar graph
ax.set_title("SVM with Anova Feature importances using MDI")       #set title
ax.set_ylabel("Mean decrease in impurity")      #set y-label
fig.tight_layout()
plt.show()


########################################## LOGISTIC CLASSIFIER ######################################################

# 1. Fit Logistic model for K = 10 and get accuracy score
Y_pred, logistic_clf = dt.Logistic_Classifier(X_train, Y_train, X_test, Y_test)
print('\nLogistic Classifier Accuracy on test data:{}'.format(round(metrics.accuracy_score(Y_test,Y_pred),3)*100))

# 2. Plot non-normalized confusion matrix
titles_options = [
    ("Logistic Confusion matrix, without normalization", None),
    #("Logistic Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        logistic_clf,
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