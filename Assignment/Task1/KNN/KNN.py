import imageio
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
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

########################################## KNN CLASSIFIER ######################################################

# #1. Fit KNN model for different values of k
# estimators = [1,20]
# accuracies_df = pd.DataFrame(list(range(1,estimators[1])), columns=["k"])
# accuracies = []
# for i in range(estimators[0],estimators[1]):
#     Y_pred_KNN = dt.KNN_Classifier(X_train, Y_train, X_test,i)
#     accuracies.append(round(accuracy_score(Y_test,Y_pred_KNN),3)*100)
#     print('Accuracy for k={} computed'.format(i))
#
# accuracies_df['accuracies']=accuracies
# print('KNN Accuracy Score on Test data:\n',accuracies_df)
# print('The value of K should be below {}'.format(round(np.sqrt(len(X_train)))))
#
# #2. Plot accuracy vs k
# fig, ax = plt.subplots()
# ax.scatter(accuracies_df['k'], accuracies_df['accuracies'])
# ax.set(title = 'Accuracy against number of neighbors K',
#             ylabel='Accuracy (%)',xlabel='Number of Estimators K', ylim=[60, 100])
# plt.title('Accuracy against number of neighbors K', weight = 'bold')
# plt.show()

# 3. Fit KNN model for K = 10 and get accuracy score
Y_pred, KNN_clf = dt.KNN_Classifier(X_train, Y_train, X_test,16)
print('KNN Accuracy Score on Test data: {}\n'.format(round(metrics.accuracy_score(Y_test,Y_pred),3)*100))

# 4. Plot non-normalized confusion matrix
titles_options = [
    ("KNN Confusion matrix, without normalization", None),
    #("KNN Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        KNN_clf,
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