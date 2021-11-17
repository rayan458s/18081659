# import libraries
import pickle
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


X, y, true_centers = make_blobs(n_samples=3000, centers=3, n_features=2, return_centers=True,
                      random_state=0)

# visualise the data
print(f"dataset shape: {X.shape}")
print(f"one of the data: {X[0,:]}, its cluster center is {true_centers[y[0]]}")
print(f"true center is: ")
print(true_centers)
#  plot datapoints according to their cluster labels
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, alpha=0.5, cmap='viridis')

#  plot true cluster centers using color 'blue'
plt.scatter(true_centers[:, 0], true_centers[:, 1], c='blue', s=200);


# sklearn functions implementation
def kmeansPredict(X, k):
    '''
    Inputs
        X: dataset;
        k: number of clusters.

    Return
        y_kmeans: predicted cluster label;
        centers: cluster centers.
    '''

    # the bulit-in function for K-means,
    # where n_clusters is the number of clusters.
    kmeans = KMeans(n_clusters=k)

    # fit the algorithm with dataset
    kmeans.fit(X)

    # predict after fit
    y_kmeans = kmeans.predict(X)

    # get the centers after fit
    centers = kmeans.cluster_centers_

    return y_kmeans, centers

# run K-means for different values of k, which is 3 in following case
k1 = 3
y_predict, centers = kmeansPredict(X,k1)

# visualise the result of k-means
print(f"The cluster center for the first datapoint is : {centers[y_predict[0]]}")
print(f"The centers for the {k1} clusters are : ")
print(centers)


#  plot datapoints according to their cluster labels
plt.scatter(X[:, 0], X[:, 1], c=y_predict, s=50, alpha=0.5, cmap='viridis')

#  plot estimate cluster centers using color 'red'
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200);

#  plot true cluster centers using color 'blue'
plt.scatter(true_centers[:, 0], true_centers[:, 1], c='blue', s=100);


# Euclidean Distance Caculator
def Euclidean_dist(a, b):
    '''
    Input
        a,b : 2 datapoints in vector form, the euclidean distance of which will be returned.
    '''
    # Complete code below.
    euc_dist = np.sum(np.square(a-b))
    return euc_dist

# test the function with two data [0,0] and [3,4]
test_data1=np.array([0,0])
test_data2=np.array([3,4])
dist=Euclidean_dist(test_data1,test_data2)
# you should see an output: 25
print(dist)



def calc_label(X,C):
    '''
    Input
        X : data matrix;
        C : cluster center matrix;


    Return
        Labels : labels based on cluster center matrix C.

    '''

    No_Data=X.shape[0]
    K=C.shape[0]

    # initial Label for each data
    Label=np.zeros(No_Data)

    # for data i we calc its distance to every center
    for i in range(No_Data):
        datum=X[i,:]
        # initial distance from datum to each center.
        dist2centers=np.zeros(K)
        for k in range(K):
            center=C[k,:]
            dist=Euclidean_dist(datum,center)
            dist2centers[k]=dist

        # get the closest label for this datum
        # Complete code below.
        Label[i]=np.argmin(dist2centers)

    return Label.astype(np.int)

# test the function with X=[[1,1],[3,3],[-1,-1],[-3,-3]], C=[[2,2],[-2,-2]]
test_X=np.array([[1,1],[3,3],[-1,-1],[-3,-3]])
test_C=np.array([[2,2],[-2,-2]])
label=calc_label(test_X,test_C)

# you should see an output: [0 0 1 1]
print(label)


def cost_func(X, C, Labels):
    '''
    Input
        X : data matrix;
        C : cluster center matrix;
        Labels : labels based on cluster center matrix C.

    Return
        cost.
    '''
    # Number of clusters
    K = C.shape[0]

    # inital cost
    cost = 0.

    # for each cluster
    for k in range(K):
        # get the idx of those data in cluster k
        data_idx = np.where(Labels==k)

        cluster_data=X[data_idx]

        cluster_center=C[k,:]

        # for each data in cluster k
        for i in range(cluster_data.shape[0]):
            # add the cost
            # Complete code below.
            cost += Euclidean_dist(cluster_data[i,:],cluster_center)

    cost = cost / K
    return cost

# test the function with X=[[1,1],[3,3],[-1,-1],[-3,-3]], C=[[2,2],[-2,-2]]
test_X=np.array([[1,1],[3,3],[-1,-1],[-3,-3]])
test_C=np.array([[2,2],[-2,-2]])
label=calc_label(test_X,test_C)
cost=cost_func(test_X, test_C, label)

# you should see an output: 8.0
print(cost)



def update_center(X, Label, K):
    '''
    Input
        X : data matrix;

        Labels : labels based on cluster center matrix C;

        K : number of clusters.

    Return
        C : cluster center matrix;
    '''

    data_dim = X.shape[1]

    # define cluster center matrix
    C = np.zeros([K, data_dim])

    # for each cluster
    for k in range(K):
        # get the idx of those data in cluster k
        data_idx = np.where(Label==k)

        cluster_data = X[data_idx]

        # calc the data mean within cluster k
        # Complete code below.
        new_cluster = np.mean(cluster_data, axis=0)
        # update the cluster center
        C[k,:] = new_cluster

    return C

# test the function with X=[[1,1],[3,3],[-1,-1],[-3,-3]], Label=[0, 0, 1, 1], K=2
test_X=np.array([[1,1],[3,3],[-1,-1],[-3,-3]])
test_Label=np.array([0, 0, 1, 1]).astype(np.int)
test_K=2
C=update_center(test_X,test_Label,test_K)

# you should see an output: [[ 2.  2.]
#                            [-2. -2.]]
print(C)



def kmeansNewPredict(X, K, max_iter):
    # Initial Centroids by randomly chosen from dataset
    No_Data=X.shape[0]
    data_idx=np.arange(No_Data)
    np.random.shuffle(data_idx)
    C = X[data_idx[:K]]

    # define list to store cost history
    cost_history = []

    # calc the initial cluster label
    Label = calc_label(X,C)

    # calc the initial cost
    cost = cost_func(X,C,Label)
    print(f"initial: cost = {cost}")
    cost_history.append(cost)

    # define a flag which indicates training or not
    Train = True
    iter = 0
    while Train:
        iter += 1

        # Complete code below.
        # update cluster center
        C = update_center(X, Label, K)

        # calc cluster label
        Label = calc_label(X,C)

        # calc cost
        cost = cost_func(X,C,Label)

        print(f"iter {iter}: cost = {cost}")
        cost_history.append(cost)

        cost_diff=cost_history[-2]-cost_history[-1]
        if iter>= max_iter:
            Train = False

    return Label, C, cost_history


# run K-means for different values of k
k2 = 3
max_iter=20
y_kmeans1, centers1, cost_history = kmeansNewPredict(X, k2, max_iter)


# print and plot out the cluster center and compare with built-in implementation
print(f"The cluster center for the first datapoint is : {centers1[y_kmeans1[0]]}")

print(f"The centers for the {k2} clusters are : ")
print(centers1)

#  plot datapoints according to their cluster labels
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans1, s=50, alpha=0.5, cmap='viridis')

#  plot cluster centers using color 'red'
plt.scatter(centers1[:, 0], centers1[:, 1], c='red', s=200);
#  plot true cluster centers using color 'blue'
plt.scatter(true_centers[:, 0], true_centers[:, 1], c='blue', s=100);