import imageio
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
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


def select_features_with_ANOVA(X,Y, k):
    fs = SelectKBest(score_func=f_classif, k=k)     # define feature selection
    X_selected = fs.fit_transform(X, Y)     # apply feature selection
    return X_selected


def reduce_dimensionality_with_PCA(X, k):
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


def get_features_importance_with_RF(X_train, Y_train):
    feature_names = [f"feature {i}" for i in range(X_train.shape[1])]       #Attribute a feature name to each vector row
    forest = RandomForestClassifier(random_state=0)     #Define the random forest classifier object
    forest.fit(X_train, Y_train)            #fit the random forest classifier model to training data
    importances = forest.feature_importances_       #get importances of each feature
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)    #Compute standard deviation of the features importances
    forest_importances = pd.Series(importances, index=feature_names)        #concatenante the features importances in a Pandas Serie
    return forest_importances, std


def KNN_Classifier(X_train, Y_train, X_test,k):
    KNN_clf = KNeighborsClassifier(n_neighbors=k)     #Create KNN object with a K coefficient
    KNN_clf.fit(X_train, Y_train) # Fit KNN model
    Y_pred_KNN = KNN_clf.predict(X_test)
    return Y_pred_KNN, KNN_clf


def Decision_Tree_Classifier(X_train, Y_train, X_test, tree_params):
    dt_clf = tree.DecisionTreeClassifier( **tree_params )      #Define the decision three classifier
    dt_clf.fit(X_train,Y_train)
    Y_pred_DT =  dt_clf.predict(X_test)
    return Y_pred_DT, dt_clf


def Random_Forest_Classifier(X_train, Y_train, X_test):
    rf_clf = RandomForestClassifier(n_estimators=100)   #Define the random forest classifier
    rf_clf.fit(X_train,Y_train)        #Train the model using the training sets
    Y_pred_RF = rf_clf.predict(X_test)      # prediction on test set
    return Y_pred_RF, rf_clf


def visualise_tree(tree_to_print):
    plt.figure()
    tree.plot_tree(tree_to_print,
               filled = True,
              rounded=True);
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=800)
    plt.show()


def Bagging_Classifier(X_train, y_train, X_test,k):
    bag_clf = BaggingClassifier(n_estimators=k,max_samples=0.5, max_features=4,random_state=1)  #Create KNN object with a K coefficient
    bag_clf.fit(X_train, y_train)      #Fit KNN model
    Y_pred_BAG = bag_clf.predict(X_test)
    return Y_pred_BAG, bag_clf


def Boosting_Classifier(X_train, Y_train, X_test,k):
    boost_clf = AdaBoostClassifier(n_estimators=k)       # AdaBoost takes Decision Tree as its base-estimator model by default.
    boost_clf.fit(X_train,Y_train,sample_weight=None)  # Fit KNN model
    Y_pred_BOOST = boost_clf.predict(X_test)
    return Y_pred_BOOST, boost_clf


def Logistic_Classifier(X_train, Y_train, X_test, Y_test):
    logistic_clf = LogisticRegression(solver='lbfgs')     # Build Logistic Regression Model
    logistic_clf.fit(X_train, Y_train)            # Train the model using the training sets
    Y_pred= logistic_clf.predict(X_test)
    return Y_pred, logistic_clf


def SVM_Classifier(X_train,Y_train, X_test, kernel):
    svm_clf = SVC(kernel=kernel)
    svm_clf.fit(X_train,Y_train)
    y_pred = svm_clf.predict(X_test)
    return y_pred, svm_clf
