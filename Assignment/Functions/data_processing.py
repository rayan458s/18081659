# Date: 17/12/2021
# Institution: University College London
# Dept: EEE
# Class: ELEC0134
# SN: 18081659
# Description: Contains all the functions to process the data and plot results


import imageio
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import csv
import cv2
import pandas as pd
from keras.utils import np_utils
import imutils
import time
from skimage.color import rgb2gray
from sklearn.model_selection import learning_curve
from matplotlib import pyplot
import sys

img_folder=r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/image'
label_file = r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/label.csv'


def plot_4_images():
    '''
    Function to plot 4 images from the Tumor Datasets
    '''
    plt.figure(figsize=(20,20))
    for i in range(4):
        file = random.choice(os.listdir(img_folder))    #select 4 random file names
        image_path= os.path.join(img_folder, file)          #create path for the dataset file
        img = np.array(imageio.imread(image_path))              #convert the image to an array
        ax=plt.subplot(2,2,i+1)                 #define the ax plot object
        ax.title.set_text(file)     #set title
        plt.imshow(img)         #show image
    plt.show()


def load_images(img_folder):
    '''
    Input:
    - 'img_folder': the folder containing all the images of tumors MRI
    Returns:
    - 'images_array': An array containing [3000images*32x32columns*3RGB] all the images (as numpy arrays)
    - 'class_name': the names of every file containing an image
    '''
    images_array=[]
    class_name=[]
    for file in sorted(os.listdir(img_folder)):     #for all the files in dataset/image
        image_path = os.path.join(img_folder, file)      #join the path to the image filename
        image = np.array(imageio.imread(image_path))             #open and convert to numpy array
        images_array.append(image)                    #final list with all the image arrays
        class_name.append(file)                             #image names
    return images_array , class_name


def load_DNN_images(img_folder, gray_2D, resize):
    '''
    Input:
    - 'img_folder': the folder containing all the images of tumors MRI
    - 'gray_2D': If not False, Convert the images from 3D grayscale to 2D grayscale
    - 'resize': If not None, used to resize the images to desired values
    Returns:
    - 'images_array': An array containing [3000images*32x32columns] all the images (as numpy arrays)
    - 'class_name': the names of every file containing an image
    '''

    images_array=[]
    class_name=[]
    for file in sorted(os.listdir(img_folder)):     #for all the files in dataset/image
        image_path = os.path.join(img_folder, file)      #join the path to the image filename
        image = np.array(imageio.imread(image_path))             #open and convert to numpy array
        if gray_2D == True:
            image = rgb2gray(image)             #Convert from 3D grayscale to 2D grayscale
            image = (image*1.0) / 255.0                   #normalise
        if resize != None:
            image = cv2.resize(image, resize)          #resize the image to desired size
        images_array.append(image)                    #final list with all the image arrays
        class_name.append(file)                             #image names
    return images_array, class_name


def load_labels(label_file_path):
    '''
    Input:
    - 'label_file_path': the file path for the csv file containing all the labels
    Returns
    - 'labels': A numpy array [3000] containing all the labels
    '''
    open_file = open(label_file_path)           #open the CSV file
    read_file = csv.reader(open_file, delimiter=',')        #read the file using the csv.reader() function
    labels = []     #list to store the labels
    for row in read_file:       #go through every row in the csv file
        if row[1] == 'no_tumor':
           labels.append(0)     #class 'no_tumor' = label 0
        else:
            labels.append(1)    #class 'tumor' = label 1
    labels.pop(0)       #remove the first row that contains a description the csv file
    size = len(labels)
    labels = np.array(labels)       #convert the list to a numpy array

    nt_counter = 0
    for label in labels:        #count the number of no_tumor classes
        if label == 0:
            nt_counter += 1
    print(f'Number of "no_tumor" labels: {nt_counter} ({round(nt_counter/size*100)}% of the dataset)')
    print(f'Number of "x_tumor" labels: {size-nt_counter} ({round(((size-nt_counter)/size)*100)}% of the dataset)')
    return labels


def load_multiclass_labels(label_file_path, one_hot):
    '''
    Input:
    - 'label_file_path': the file path for the csv file containing all the labels
    - 'one_hot': True to apply one hot encoding to the labels, False otherwise
    Returns
    - 'labels': A numpy array [3000] containing all the labels (type of tumors)
    '''
    open_file = open(label_file_path)           #open the CSV file
    read_file = csv.reader(open_file, delimiter=',')        #read the file using the csv.reader() function
    labels = []     #list to store the labels
    for row in read_file:       #go through every row in the csv file
        labels.append(row[1])
    labels.pop(0)       #remove the first row that contains a description the csv file
    labels = np.array(labels)       #convert the list to a numpy array
    print('\nThe classes are: ',np.unique(labels))
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_Y = encoder.transform(labels)           # encode class values as integer
    dummy_y = np_utils.to_categorical(encoded_Y)    # convert integers to dummy variables (i.e. one hot encoded)
    if one_hot == True:
        labels = dummy_y
    else:
        labels = encoded_Y
    return labels


def image_array_to_vector(images_array):
    '''
    Input:
    - 'images_array': the numpy array containing all the images as arrays
    Returns
    - 'image_vectors': a numpy array containing all the images as vectors of pixel intensities
    '''
    image_vectors = []
    for i in range(len(images_array)):      #for every image array
        image = images_array[i]
        image_vector = image.flatten()        #Flatten the image into a list of raw pixel intensities
        image_vectors.append(image_vector)
    image_vectors = np.array(image_vectors)
    return image_vectors


def process_data():
    '''
    Return
    - 'images_vectors': a numpy array containing all the images as vectors of pixel intensities
    - 'labels': a list with th dataset label
    '''
    #Get images (inputs) array
    images_array, class_name = load_images(img_folder)
    images_array = np.array(images_array)

    print("\nDataset shape: {}".format(images_array.shape))
    a,b,c,d = images_array.shape
    print("Image Size: {}x{}x{}".format(b,c,d))
    print("Number of Images: {}".format(a))

    #Array to  Vectors
    images_vectors = image_array_to_vector(images_array)
    print("Vector Size: {}".format(len(images_vectors[0])))

    #Get labels (outputs) array
    labels = load_labels(label_file)
    print("Number of Labels: {}".format(len(labels)))

    return images_vectors, labels


def process_multiclass_data(one_hot=False, gray_2D=False, resize=None):
    '''
    Input:
    - 'one_hot': If True,  apply one hot encoding to the labels, False otherwise
    - 'gray_2D': If True, Convert the images from 3D grayscale to 2D grayscale
    - 'resize': If not None, used to resize the images to desired values
    Return
    - 'images_array': a numpy array containing all the images as arrays
    - 'images_vectors': a numpy array containing all the images as vectors of pixel intensities
    - 'labels': a list with th dataset label
    '''
    #Get images (inputs) array
    images_array, class_name = load_DNN_images(img_folder, gray_2D, resize)
    images_array = np.array(images_array)

    print("Dataset shape: {}".format(images_array.shape))
    if gray_2D == True:
        a,b,c = images_array.shape
        print("Image Size: {}x{}".format(b,c))
    else:
        a,b,c,d = images_array.shape
        print("Image Size: {}x{}x{}".format(b,c,d))
    print("Number of Images: {}".format(a))

    #Array to  Vectors
    images_vectors = image_array_to_vector(images_array)
    print("Vector Size: {}".format(len(images_vectors[0])))

    #Get labels (outputs) array
    labels = load_multiclass_labels(label_file, one_hot)
    print("Number of Labels: {}".format(len(labels)))

    return images_array, images_vectors, labels


def process_test_data():
    '''
    Return
    - 'images_array': a numpy array containing all the test images as arrays
    - images_vectors: a numpy array containing all the test images as vectors of pixel intensities
    - labels: a list with test labels
    '''
    #Get images (inputs) array
    images_array, class_name = load_images(r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/image_test')
    images_array = np.array(images_array)

    print("Dataset shape: {}".format(images_array.shape))
    a,b,c,d = images_array.shape
    print("Image Size: {}x{}x{}".format(b,c,d))
    print("Number of Images: {}".format(a))

    #Array to  Vectors
    images_vectors = image_array_to_vector(images_array)
    print("Vector Size: {}".format(len(images_vectors[0])))

    #Get labels (outputs) array
    labels = load_labels(r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/label_test.csv')
    print("Number of Labels: {}\n".format(len(labels)))

    return images_array, images_vectors, labels


def process_multiclass_test_data(one_hot=False, gray_2D=False, resize=None):
    '''
    Input:
    - 'one_hot': If True,  apply one hot encoding to the labels, False otherwise
    - 'gray_2D': If True, Convert the images from 3D grayscale to 2D grayscale
    - 'resize': If not None, used to resize the images to desired values
    Return
    - 'images_array': a numpy array containing all the test images as arrays
    - images_vectors: a numpy array containing all the test images as vectors of pixel intensities
    - labels: a list with test labels
    '''
    #Get images (inputs) array
    img_folder = r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/image_test'
    images_array, class_name = load_DNN_images(img_folder, gray_2D, resize)
    images_array = np.array(images_array)

    print("Dataset shape: {}".format(images_array.shape))
    if gray_2D == True:
        a,b,c = images_array.shape
        print("Image Size: {}x{}".format(b,c))
    else:
        a,b,c,d = images_array.shape
        print("Image Size: {}x{}x{}".format(b,c,d))
    print("Number of Images: {}".format(a))

    #Array to  Vectors
    images_vectors = image_array_to_vector(images_array)
    print("Vector Size: {}".format(len(images_vectors[0])))

    #Get labels (outputs) array
    label_file = r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/label_test.csv'
    labels = load_multiclass_labels(label_file, one_hot)
    print("Number of Labels: {}".format(len(labels)))

    return images_array, images_vectors, labels

def select_features_with_ANOVA(images_vectors,labels, k):
    '''
    Inputs
    - 'images_vectors': dataset images vectors (pixel intensities)
    - 'labels': dataset labels
    - 'k': number of features we want to keep
    Return
    - 'images_features': a numpy array with all the images as vectors of new K best ANOVA selected features
    '''
    fs = SelectKBest(score_func=f_classif, k=k)     # define feature selection
    images_features = fs.fit_transform(images_vectors, labels)     # apply feature selection
    return images_features


def reduce_dimensionality_with_PCA(images_vectors, k):
    '''
    Inputs
    - 'images_vectors': dataset images vectors (pixel intensities)
    - 'k': number of Components
    Return
    - 'images_features': a numpy array with all the images as vectors of new PCA projected features
    '''
    pca = PCA(n_components=k)   # the built-in function for PCA, where n_clusters is the number of clusters.
    pca.fit(images_vectors)      # fit the algorithm with dataset
    Vcomp = pca.components_ #The estimated number of components (eigenvectors)

    images_features = []
    single_image_feature = []
    for image_vector in images_vectors:     #go through all the image vectors (pixel intensities)
        for component in Vcomp:         #go through all the eigenvectors computed using PCA
            single_image_feature.append(abs(np.dot(image_vector,component)))        #perform dot product between the each eigenvector and each image vector
        images_features.append(single_image_feature)
        single_image_feature = []
    return np.array(images_features)


def process_ANOVA_features(Task, k_ANOVA):
    '''
    Inputs
    - 'Task': 1 or 2. Used to know what file to store the feature
    - k_ANOVA: The number of features we want to keep
    Return
        ANOVA_features: A numpy array containing every image as a vector of ANOVA selected features
    '''
    if Task == 1:
        images_vectors, labels = process_data()         #Select k Features using ANOVA
    elif Task == 2:
        images_array, images_vectors, labels = process_multiclass_data()         #Select k Features using ANOVA
    start_time = time.time()        #start the time counter
    ANOVA_features = select_features_with_ANOVA(images_vectors, labels, k_ANOVA)        #reduce dimensionality using ANOVA (supervised technique)
    elapsed_time = time.time() - start_time     #get the elapsed time since the counter started
    print(f"\nElapsed time to select features using ANOVA: {elapsed_time/60:.2f} minutes")
    print("Selected number of features: {}".format(k_ANOVA))
    print("Final Input Data Shape: {}".format(np.array(ANOVA_features).shape))
    file_path = '/Users/rayan/PycharmProjects/AMLS/Assignment/Task'+str(Task)+'/Features/ANOVA-'+str(k_ANOVA)+'.dat'    #file path tos ave the computed features
    np.savetxt(file_path, ANOVA_features)       #save the computed features for re-use
    return ANOVA_features


def process_PCA_features(Task, k_PCA):
    '''
    Inputs
    - 'Task': 1 or 2. Used to know what file to store the feature
    - 'k_PCA': The number of features we want to keep
    Return
    - 'PCA_features': A numpy array containing every image as a vector of PCA projected features
    '''
    if Task == 1:
        images_vectors, labels = process_data()         #Select k Features using ANOVA
    elif Task == 2:
        images_array, images_vectors, labels = process_multiclass_data()         #Select k Features using ANOVA
    start_time = time.time()        #start the time counter
    PCA_features = reduce_dimensionality_with_PCA(images_vectors, k_PCA)        #reduce dimensionality using ANOVA (unsupervised technique)
    elapsed_time = time.time() - start_time     #get the elapsed time since the counter started
    print(f"\nElapsed time to select features using PCA: {elapsed_time/60:.2f} minutes")
    print("Selected number of features: {}".format(k_PCA))
    print("Final Input Data Shape: {}".format(np.array(PCA_features).shape))
    file_path = '/Users/rayan/PycharmProjects/AMLS/Assignment/Task'+str(Task)+'/Features/PCA-'+str(k_PCA)+'.dat'    #file path tos ave the computed features
    np.savetxt(file_path, PCA_features)         ##save the computed features for re-use
    return PCA_features


def get_features(file_path):
    '''
    Inputs
    - 'file_path': The filepath of the features
    Return
    - 'features': A numpy array containing every image as a vector of PCA/ANOVA features
    '''
    features = np.loadtxt(file_path)      #load the features file
    print("Selected number of features: {}".format(len(features[0])))
    print("Final Input Data Shape: {}".format(np.array(features).shape))
    return features


def get_features_importance_with_RF(X_train, Y_train):
    '''
    Inputs
    - 'X_train': training images vectors of selected features
    - 'Y_train': training labels
    Return
    - 'forest_importances': Panda Seroe with the  features importances values
    - 'std': the standard deviation of the feature importances
    '''
    feature_names = [f"feature {i}" for i in range(X_train.shape[1])]       #Attribute a feature name to each vector row
    forest = RandomForestClassifier(random_state=0)     #Define the random forest classifier object
    forest.fit(X_train, Y_train)            #fit the random forest classifier model to training data
    importances = forest.feature_importances_       #get importances of each feature
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)    #Compute standard deviation of the features importances
    forest_importances = pd.Series(importances, index=feature_names)        #concatenante the features importances in a Pandas Serie
    return forest_importances, std


def plot_features_importances(X_train, Y_train, dim_reduction, n_features):
    '''
    Function to plot the features importances
    Inputs
    - 'X_train': training images vectors of selected features
    - 'Y_train': training labels
    - 'dim_reduction': PCA/ANOVA
    - 'n_features': Number of features selected
    '''
    forest_importances, std = get_features_importance_with_RF(X_train, Y_train)
    fig, ax = plt.subplots()            #define the plot object
    forest_importances.plot.bar(yerr=std, ax=ax)        #plot bar graph
    ax.set_title(f"{dim_reduction} ({n_features}) Feature Importances Using MDI")       #set title
    ax.set_ylabel("Mean decrease in impurity")      #set y-label
    fig.tight_layout()
    plt.show()



def plot_learning_curve(estimator, title, X, y, scoring =None, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Generate the test and training learning curve
    Extracted and modified from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py.
    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        fig, ax = plt.subplots()

    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")

    train_sizes,train_scores,test_scores,fit_times,_ = learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes,return_times=True, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r",)
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g",)
    ax.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training Loss")
    ax.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation Accuracy")
    ax.legend(loc="best")
    return plt


def plot_loss_accuracy(history, val=True):
    """
    Generate CNN learning curve
    Extracted and modified from: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
    - 'history': trained CNN
    """
    # plot diagnostic learning curves
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='Training')
    if val == True:
        plt.plot(history.history['val_loss'], color='orange', label='Validation')
    plt.legend(loc='upper right')
    plt.grid(visible = True)

    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='Training')
    if val == True:
        plt.plot(history.history['val_accuracy'], color='orange', label='Validation')
    plt.legend(loc='lower right')
    plt.grid(visible = True)
    plt.show()
