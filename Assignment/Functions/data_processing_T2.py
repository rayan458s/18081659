import imageio
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import csv
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import imutils
import time


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
    Input: the folder containing all the images of tumors MRI
    Returns
    - 'images_array': An array containing [3000images*512rows*512columns*3RGB] all the images (as numpy arrays)
    - 'class_name': the names of every file containing an image
    '''
    images_array=[]
    class_name=[]
    for file in os.listdir(img_folder):     #for all the files in dataset/image
        image_path = os.path.join(img_folder, file)      #join the path to the image filename
        image = np.array(imageio.imread(image_path))             #open and convert to numpy array
        images_array.append(image)                    #final list with all the image arrays
        class_name.append(file)                             #image names
    return images_array , class_name


def load_labels(label_file_path):
    '''
    Input: the file path for the csv file containing all the labels
    Returns 'labels': A numpy array [3000] containing all the labels (type of tumors)
    '''
    open_file = open(label_file_path)           #open the CSV file
    read_file = csv.reader(open_file, delimiter=',')        #read the file using the csv.reader() function
    labels = []     #list to store the labels
    for row in read_file:       #go through every row in the csv file
        labels.append(row[1])
    labels.pop(0)       #remove the first row that contains a description the csv file
    labels = np.array(labels)       #convert the list to a numpy array
    encoder = LabelEncoder()        # encode class values as integer
    encoder.fit(labels)
    encoded_Y = encoder.transform(labels)
    dummy_y = np_utils.to_categorical(encoded_Y)    # convert integers to dummy variables (i.e. one hot encoded)
    labels = dummy_y
    return labels


def image_array_to_vector(images_array, size=(512, 512)):
    '''
    Input: the numpy array containing all the images as arrays
    Returns 'image_vectors': a numpy array containing all the images as vectors of pixel intensities
    '''
    image_vectors = []
    for i in range(len(images_array)):      #for every image array
        image = images_array[i]
        image_vector = cv2.resize(image, size).flatten()        # resize the image to a fixed size (not modified)
        image_vectors.append(image_vector)      #Flatten the image into a list of raw pixel intensities
    image_vectors = np.array(image_vectors)
    return image_vectors


def process_data():
    '''
    Return
        images_vectors: a numpy array containing all the images as vectors of pixel intensities
        labels: a list with th dataset label
    '''
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

    return images_vectors, labels, images_array

def process_data_CNN():
    '''
    Return
        images_vectors: a numpy array containing all the images as vectors of pixel intensities
        labels: a list with th dataset label
    '''
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

    return images_array, labels, images_array


def select_features_with_ANOVA(images_vectors,labels, k):
    '''
    Inputs
        images_vectors: dataset images vectors (pixel intensities)
        labels: dataset labels
        k: number of features we want to keep
    Return
        images_features: a numpy array with all the images as vectors of new K best ANOVA selected features
    '''
    fs = SelectKBest(score_func=f_classif, k=k)     # define feature selection
    images_features = fs.fit_transform(images_vectors, labels)     # apply feature selection
    return images_features


def reduce_dimensionality_with_PCA(images_vectors, k):
    '''
    Inputs
        images_vectors: dataset images vectors (pixel intensities)
        k: number of Components
    Return
        images_features: a numpy array with all the images as vectors of new PCA projected features
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


def process_ANOVA_features(k_ANOVA):
    '''
    Inputs
        k_ANOVA: The number of features we want to keep
    Return
        ANOVA_features: A numpy array containing every image as a vector of ANOVA selected features
    '''
    images_vectors, labels = process_data()         #Select k Features using ANOVA
    start_time = time.time()        #start the time counter
    ANOVA_features = select_features_with_ANOVA(images_vectors, labels, k_ANOVA)        #reduce dimensionality using ANOVA (supervised technique)
    elapsed_time = time.time() - start_time     #get the elapsed time since the counter started
    print(f"\nElapsed time to select features using ANOVA: {elapsed_time/60:.2f} minutes")
    print("\nSelected number of features: {}".format(k_ANOVA))
    print("\nFinal Input Data Shape: {}".format(np.array(ANOVA_features).shape))
    file_path = '/Users/rayan/PycharmProjects/AMLS/Assignment/Task2/Features/ANOVA-'+str(k_ANOVA)+'.dat'    #file path tos ave the computed features
    np.savetxt(file_path, ANOVA_features)       #save the computed features for re-use
    return ANOVA_features


def process_PCA_features(k_PCA):
    '''
    Inputs
        k_PCA: The number of features we want to keep
    Return
        PCA_features: A numpy array containing every image as a vector of PCA projected features
    '''
    images_vectors, labels = process_data()     #Project the data into 10 Features using PCA
    start_time = time.time()        #start the time counter
    PCA_features = reduce_dimensionality_with_PCA(images_vectors, k_PCA)        #reduce dimensionality using ANOVA (unsupervised technique)
    elapsed_time = time.time() - start_time     #get the elapsed time since the counter started
    print(f"\nElapsed time to select features using PCA: {elapsed_time/60:.2f} minutes")
    print("\nSelected number of features: {}".format(k_PCA))
    print("\nFinal Input Data Shape: {}".format(np.array(PCA_features).shape))
    file_path = '/Users/rayan/PycharmProjects/AMLS/Assignment/Task2/Features/PCA-'+str(k_PCA)+'.dat'   #file path tos ave the computed features
    np.savetxt(file_path, PCA_features)         ##save the computed features for re-use
    return PCA_features


def get_ANOVA_features(k_ANOVA):
    '''
    Inputs
        k_ANOVA: The number of features we want to keep
    Return
        ANOVA_features: A numpy array containing every image as a vector of PCA projected features
    '''
    file_path = '/Users/rayan/PycharmProjects/AMLS/Assignment/Task2/Features/ANOVA-'+str(k_ANOVA)+'.dat'      #get filepath where we saved the previously computed features
    ANOVA_features = np.loadtxt(file_path)      #load the features file
    print("\nSelected number of features: {}".format(len(ANOVA_features[0])))
    print("\nFinal Input Data Shape: {}".format(np.array(ANOVA_features).shape))
    return ANOVA_features


def get_PCA_features(k_PCA):
    '''
    Inputs
        k_PCA: The number of features we want to keep
    Return
        PCA_features: A numpy array containing every image as a vector of PCA projected features
    '''
    file_path = '/Users/rayan/PycharmProjects/AMLS/Assignment/Task2/Features/PCA-'+str(k_PCA)+'.dat'   #get filepath where we saved the previously computed features
    PCA_features = np.loadtxt(file_path)         #load the features file
    print("\nSelected number of features: {}".format(len(PCA_features[0])))
    print("\nFinal Input Data Shape: {}".format(np.array(PCA_features).shape))
    return PCA_features


def get_features_importance_with_RF(X_train, Y_train):
    '''
    Inputs
        X_train: training images vectors of selected features
        Y_train: training labels
    Return
        forest_importances: Panda Seroe with the  features importances values
        std: the standard deviation of the feature importances
    '''
    feature_names = [f"feature {i}" for i in range(X_train.shape[1])]       #Attribute a feature name to each vector row
    forest = RandomForestClassifier(random_state=0)     #Define the random forest classifier object
    forest.fit(X_train, Y_train)            #fit the random forest classifier model to training data
    importances = forest.feature_importances_       #get importances of each feature
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)    #Compute standard deviation of the features importances
    forest_importances = pd.Series(importances, index=feature_names)        #concatenante the features importances in a Pandas Serie
    return forest_importances, std


def plot_features_importances(X_train, Y_train, dim_reduction, n_features):
    forest_importances, std = get_features_importance_with_RF(X_train, Y_train)
    fig, ax = plt.subplots()            #define the plot object
    forest_importances.plot.bar(yerr=std, ax=ax)        #plot bar graph
    ax.set_title(f"{dim_reduction} ({n_features}) Feature Importances Using MDI")       #set title
    ax.set_ylabel("Mean decrease in impurity")      #set y-label
    fig.tight_layout()
    plt.show()


