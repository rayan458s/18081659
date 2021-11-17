import pickle
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



X1, Y1 = make_blobs(n_samples=3000, centers=4, n_features=3,random_state=0)

# visualise the data
print("dataset shape:" ,X1.shape)
#  plot datapoints according to their cluster labels
plt.scatter(X1[:, 0], X1[:, 1], c=Y1, s=50, alpha=0.5, cmap='viridis')

#  plot true cluster centers using color 'blue'
#plt.scatter(true_centers[:, 0], true_centers[:, 1], c='blue', s=200);


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

    # the bulit-in function for PCA,
    # where n_clusters is the number of clusters.
    pca = PCA(n_components=k)

    # fit the algorithm with dataset
    pca.fit(X)

    Variance = pca.explained_variance_ratio_
    SValue = pca.singular_values_
    Vcomp = pca.components_
    return SValue, Variance, Vcomp

# run PCA for different values of k, which is 3 in following case
k1 = 2
SingularValue, Variance, Vcomponent = PCAPredict(X1,k1)
print(SingularValue)
print(Variance)
print(Vcomponent)


def PCANewPredict(X , num_components):

    #Substract means
    X_meaned = X - np.mean(X , axis = 0)

    #Compute the Covariance  Matrix
    cov =  np.mean(np.dot(X_meaned, X_meaned.T))
    #Compute Eigen value and vectors
    eigen_values , eigen_vectors = np.linalg.eigh(cov)

    #Sort Eigen value and vectors
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_vecs = np.argsort(eigen_vectors)[::-1]

    #Choose num_components of first sorted eigenvector
    num_components = sorted_vecs[0]
    #Compute the dimension reduced datapoints
    X_reduced = np.dot(num_components.T,X_meaned.T)
    return X_reduced

# run PCA for different values of k
k2 = 2
Xreduced = PCANewPredict(X1, k2)
print(Xreduced.shape)