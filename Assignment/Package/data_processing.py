import imageio
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
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import csv
import cv2
import pandas as pd
import imutils
import time


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


def reduce_dimensionality_with_PCA(images_vectors, k):
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
    pca.fit(images_vectors)      # fit the algorithm with dataset

    Variance = pca.explained_variance_ratio_
    SValue = pca.singular_values_
    Vcomp = pca.components_

    images_features = []
    single_image_feature = []
    for image_vector in images_vectors:
        for component in Vcomp:
            single_image_feature.append(abs(np.dot(image_vector,component)))
        images_features.append(single_image_feature)
        single_image_feature = []
    return np.array(images_features)


def get_features_importance_with_RF(X_train, Y_train):
    feature_names = [f"feature {i}" for i in range(X_train.shape[1])]       #Attribute a feature name to each vector row
    forest = RandomForestClassifier(random_state=0)     #Define the random forest classifier object
    forest.fit(X_train, Y_train)            #fit the random forest classifier model to training data
    importances = forest.feature_importances_       #get importances of each feature
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)    #Compute standard deviation of the features importances
    forest_importances = pd.Series(importances, index=feature_names)        #concatenante the features importances in a Pandas Serie
    return forest_importances, std

def extract_color_histogram(images, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
    features_vectors = []
    for image in images:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])
        # handle normalizing the histogram if we are using OpenCV 2.4.X
        if imutils.is_cv2():
            hist = cv2.normalize(hist)
        # otherwise, perform "in place" normalization in OpenCV 3 (I
        # personally hate the way this is done
        else:
            cv2.normalize(hist, hist)
        # return the flattened histogram as the feature vector
    features_vectors.append(hist.flatten())
    return features_vectors


def process_data():
    #Get images (inputs) array
    images_array, class_name = load_images(img_folder)
    images_array = np.array(images_array)

    print("\nDataset shape: {}".format(images_array.shape))
    a,b,c,d = images_array.shape
    print("\nImage Size: {}x{}x{}".format(b,c,d))
    print("\nNumber of Images: {}".format(a))

    #Get labels (outputs) array
    labels = load_labels(label_file)
    print("\nNumber of Labels: {}".format(len(labels)))

    #Array to  Vectors
    images_vectors = image_array_to_vector(images_array)
    print("\nVector Size: {}".format(len(images_vectors[0])))

    return images_vectors, labels


def process_ANOVA_features(k_ANOVA):
    images_vectors, labels = process_data()#Select 10 Features using ANOVA
    start_time = time.time()
    ANOVA_features = select_features_with_ANOVA(images_vectors, labels, k_ANOVA)
    elapsed_time = time.time() - start_time
    print(f"\nElapsed time to select features using ANOVA: {elapsed_time/60:.2f} minutes")
    print("\nSelected number of features: {}".format(k_ANOVA))
    print("\nFinal Input Data Shape: {}".format(np.array(ANOVA_features).shape))
    np.savetxt('/Users/rayan/PycharmProjects/AMLS/Assignment/Task1/Dim-Reduction/ANOVA.dat', ANOVA_features)
    return ANOVA_features


def process_PCA_features(k_PCA):
    images_vectors, labels = process_data()#Select 10 Features using ANOVA
    start_time = time.time()
    PCA_features = reduce_dimensionality_with_PCA(images_vectors, k_PCA)
    elapsed_time = time.time() - start_time
    print(f"\nElapsed time to select features using PCA: {elapsed_time/60:.2f} minutes")
    print("\nSelected number of features: {}".format(k_PCA))
    print("\nFinal Input Data Shape: {}".format(np.array(PCA_features).shape))
    np.savetxt('/Users/rayan/PycharmProjects/AMLS/Assignment/Task1/Dim-Reduction/PCA.dat', PCA_features)
    return PCA_features


def get_ANOVA_features():
    ANOVA_features = np.loadtxt('/Users/rayan/PycharmProjects/AMLS/Assignment/Task1/Dim-Reduction/ANOVA-10.dat')
    return ANOVA_features


def get_pca_features():
    PCA_features = np.loadtxt('/Users/rayan/PycharmProjects/AMLS/Assignment/Task1/Dim-Reduction/ANOVA-10.dat')
    return PCA_features


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
