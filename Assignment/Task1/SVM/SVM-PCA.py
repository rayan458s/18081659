import imageio
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import ConfusionMatrixDisplay
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


img_folder=r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/image_small'
label_file = r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/label_small.csv'

from Assignment.Package import data_processing as dt


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

#Dimensionality Reduction using PCA (feature projection)
k_PCA = 10
SingularValue, Variance, Vcomponent = dt.reduce_dimensionality_with_PCA(images_vectors,k_PCA)
images_features = []
single_image_feature = []
for image_vector in images_vectors:
    for component in Vcomponent:
        single_image_feature.append(abs(np.dot(image_vector,component)))
    images_features.append(single_image_feature)
    single_image_feature = []
print("\nSelected number of features: {}".format(k_PCA))

#Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(images_features,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}'.format(round(((len(Y_train)*1.0)/len(images_features)),3),round((len(Y_test)*1.0)/len(labels),3)))

#Plot the features importances
forest_importances, std = dt.get_features_importance_with_RF(X_train, Y_train)
fig, ax = plt.subplots()            #define the plot object
forest_importances.plot.bar(yerr=std, ax=ax)        #plot ar graph
ax.set_title("SVM with PCA Feature importances using MDI")       #set title
ax.set_ylabel("Mean decrease in impurity")      #set y-label
fig.tight_layout()
plt.show()

########################################## SVM CLASSIFIER ######################################################

# #1. Test SVM model accuracy for different Kernel
# kernels = ["linear", "rbf"]
# accuracies_df = pd.DataFrame(list(range(1,3)), columns=["kernels"])
# accuracies = []
# for kernel in kernels:
#     Y_pred_SVM, svm_clf = dt.SVM_Classifier(X_train, Y_train, X_test,kernel)
#     accuracies.append(round(accuracy_score(Y_test,Y_pred_SVM),2)*100)
#     print('\nAccuracy for kernel = {} computed'.format(kernel))
# accuracies_df['accuracies']=accuracies

# #2. Plot accuracy vs kernel type
# fig, ax = plt.subplots()
# ax.scatter(accuracies_df['kernels'], accuracies_df['accuracies'])
# ax.set(title = 'Accuracy against type of Kernel',
#             ylabel='Accuracy (%)',xlabel='Kernel', ylim=[60, 100])
# plt.title('Accuracy against type of Kernel', weight = 'bold')
# plt.show()

# 3. Fit SVM model for linear Kernel and get accuracy score
Y_pred, svm_clf = dt.SVM_Classifier(X_train, Y_train, X_test,"linear")
print('\nSVM with PCA Accuracy Score on Test data: {}\n'.format(round(metrics.accuracy_score(Y_test,Y_pred),3)*100))

# 4. Plot non-normalized confusion matrix
titles_options = [
    ("SVM (with PCA) Confusion matrix, without normalization", None),
    #("SVM Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        svm_clf,
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