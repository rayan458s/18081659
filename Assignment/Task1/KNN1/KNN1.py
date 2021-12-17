# Date: 17/12/2021
# Institution: University College London
# Dept: EEE
# Class: ELEC0134
# SN: 18081659
# Description: KNN Testing. See plots file for Results

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys

from Assignment.Functions import data_processing as dt
from Assignment.Functions import Classifiers as classf


########################################## DATA PROCESSING ######################################################

# 1. Get the input data (matrix of image vectors and output data (vector of labels)
images_vectors, labels = dt.process_data()      #process all the images into pixel vectors and get the labels

# 2. Transform inputs into feature vectors
n_features = 10     #define the number of features to select/project from the pixels vectors
dim_reduction = 'PCA'       #define the DR method
filepath = '/Users/rayan/PycharmProjects/AMLS/Assignment/Task1/Features/'+dim_reduction+'-'+str(n_features)+'.dat'  #get corresponding filepath
images_features = dt.get_features(filepath)     #project the desired number of feartures using PCA

# 3. Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(images_features,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}\n'.format(round(((len(Y_train)*1.0)/len(images_features)),3),round((len(Y_test)*1.0)/len(labels),3)))


########################################## KNN CLASSIFIER ######################################################

#1. Number of Nearest Neighbors Tunning
KNN_scores_df = pd.DataFrame(list(range(1,16)), columns=["k"])      #create pandas dataframe to containe the performance metrics for difference values of K
accuracies = []
precisions = []
recalls = []
estimators_range = [1, 16]      #the range to test the number of nearest neighbors
for i in range(estimators_range[0],estimators_range[1]):        #classify the data for every value of K and get accuracy, precision and recall metrics
    Y_pred, KNN_clf = classf.KNN_Classifier(X_train, Y_train, X_test,i)
    accuracies.append(round(metrics.accuracy_score(Y_test,Y_pred),2)*100)
    precisions.append(round(metrics.precision_score(Y_test,Y_pred),2)*100)
    recalls.append(round(metrics.recall_score(Y_test,Y_pred),2)*100)

KNN_scores_df['accuracies']=accuracies      #create new column in the dataframe with the accuracies
KNN_scores_df['precisions']=precisions         #create new column in the dataframe with the precisions
KNN_scores_df['recalls']=recalls                    #create new column in the dataframe with the recalls
print(f"\nKNN ({dim_reduction} {n_features}) performance:\n")
print(KNN_scores_df)

#2. Estimators Visualisation
# visualise the performance metrics vs the number of nearest neighbors in a plot
fig, ax = plt.subplots()
ax.scatter(KNN_scores_df['k'], KNN_scores_df['accuracies'], c='b', label='Accuracy')
ax.scatter(KNN_scores_df['k'], KNN_scores_df['precisions'], c='g', label='Precision')
ax.scatter(KNN_scores_df['k'], KNN_scores_df['recalls'], c='r', label='Recall')
ax.set(ylabel='Performance (%)',xlabel='Number of Estimators K', ylim=[50, 110])
plt.legend(loc='lower right')
plt.grid(visible = True)
plt.title(f'KNN ({dim_reduction} {n_features}) Performance vs Number of Estimators K', weight = 'bold')
plt.show()


# 3. Fit KNN model for best hyperparameter
K = 1
start_time = time.time()        #start the time counter (to determine the time taken to classify the data
Y_pred_KNN, KNN_clf = classf.KNN_Classifier(X_train, Y_train, X_test, K)         #classify the data using the KNN Classifier
elapsed_time = time.time() - start_time     #get the elapsed time since the counter was started
print(f"\nElapsed time to classify the data using KNN ({dim_reduction} {n_features}) Classifier for K = {K} {elapsed_time:.2f} seconds")

# 4. Get Performance Scores
KNN_accuracy = round(accuracy_score(Y_test,Y_pred_KNN),2)*100     #get accuracy
KNN_precision = round(precision_score(Y_test,Y_pred_KNN),2)*100       #get precision
KNN_recall = round(recall_score(Y_test,Y_pred_KNN),2)*100               #get recall
print(f'\nKNN ({dim_reduction} {n_features}) Accuracy Score on Test data: {KNN_accuracy}%')
print(f'\nKNN ({dim_reduction} {n_features}) Precision Score on Test data: {KNN_precision}%')
print(f'\nKNN ({dim_reduction} {n_features}) Recall Score on Test data: {KNN_recall}%')

# 5. Plot non-normalized confusion matrix
titles_options = [
    (f"KNN ({dim_reduction} {n_features}) Confusion matrix for K = {K}", None),
    #("KNN ({dim_reduction}) Normalized confusion matrix", "true"),
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
plt.show()

