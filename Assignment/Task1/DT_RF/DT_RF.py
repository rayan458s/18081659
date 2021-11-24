import imageio
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import csv
import cv2
import pandas as pd
import time

from Assignment.Package import data_processing as dt

img_folder=r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/image_small'
label_file = r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/label_small.csv'


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
print('\ntrain set: {}  | test set: {}\n'.format(round(((len(Y_train)*1.0)/len(images_vectors)),3),round((len(Y_test)*1.0)/len(labels),3)))


########################################## DT CLASSIFIER ######################################################

# 1. Fit Decision Three model and get accuracy
tree_params={'criterion':'entropy'}
start_time = time.time()
Y_pred_DT, dt_clf = dt.Decision_Tree_Classifier(X_train, Y_train, X_test, tree_params)
elapsed_time = time.time() - start_time
print(f"Elapsed time to classify the data using Regular Decision Three Classifier: {elapsed_time/60:.2f} minutes")
DT_accuracy = round(accuracy_score(Y_test,Y_pred_DT),2)*100
print('\nRegular Decision Tree Accuracy Score on Test data: {}%\n'.format(DT_accuracy))

# 2. Decision Three visualisation
dt.visualise_tree(dt_clf)

# 3. Plot non-normalized confusion matrix
titles_options = [
    ("Decision Three Confusion matrix", None),
    #("Decision Three Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        dt_clf,
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


# 4. Hyperparameter Tuning
tree_params = {'criterion': 'entropy', 'min_samples_split':50}
start_time = time.time()
Y_pred_DT_2, dt_clf_2 = dt.Decision_Tree_Classifier(X_train, Y_train, X_test, tree_params)
DT_accuracy_2 = round(accuracy_score(Y_test,Y_pred_DT),2)*100
elapsed_time = time.time() - start_time
print(f"Elapsed time to classify the data using Regular Decision Three Classifier wth tuned hyperparameters : {elapsed_time/60:.2f} minutes")
print('\nRegular Decision Tree after Hyperparameters Tuning Accuracy Score on Test data: {}%'.format(DT_accuracy_2))

# 5. Decision Three visualisation after tuning
dt.visualise_tree(dt_clf_2)

# 6. Plot non-normalized confusion matrix after tuning
titles_options = [
    ("Regular Decision Three Confusion matrix", None),
    #(" Decision Three Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        dt_clf,
        X_test,
        Y_test,
        display_labels=["No Tumor", "Tumor"],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
plt.show()

########################################## RF CLASSIFIER ######################################################

# 1. Fit Random Forest model and get accuracy score
start_time = time.time()
Y_pred_RF, rf_clf = dt.Random_Forest_Classifier(X_train, Y_train, X_test)
RF_accuracy = round(accuracy_score(Y_test, Y_pred_RF),2)*100
elapsed_time = time.time() - start_time
print(f"Elapsed time to classify the data using Regular Random Forest Classifier: {elapsed_time/60:.2f} minutes")
print("\nRegular Random Forest Accuracy Score on Test data: {}%".format(RF_accuracy))

# 2. Random Forest  visualisation
for index in range(0, 5):
    dt.visualise_tree(rf_clf.estimators_[index])

# 3. Plot non-normalized confusion matrix
titles_options = [
    ("Regular Random Forest Confusion matrix", None),
    #("Random Forest Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        rf_clf,
        X_test,
        Y_test,
        display_labels=["No Tumor", "Tumor"],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
plt.show()

# 4. Remove unimportant features + retrain and re-visualise