# Date: 17/12/2021
# Institution: University College London
# Dept: EEE
# Class: ELEC0134
# SN: 18081659
# Description: KNN with Bagging Testing. See plots file for results

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import time
import pandas as pd
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

#3. Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(images_features,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}\n'.format(round(((len(Y_train)*1.0)/len(images_features)),3),round((len(Y_test)*1.0)/len(labels),3)))


########################################## BAGGING ######################################################

#1. Estimators Tunning
BAG_scores_df = pd.DataFrame(list(range(1,16)), columns=["k"])  #create pandas dataframe to containe the performance metrics for difference values of K
accuracies = []
precisions = []
recalls = []
estimators_range = [1, 16]      #the range to test the number of nearest neighbors
for i in range(estimators_range[0],estimators_range[1]):    #classify the data for every value of K using bagging and get accuracy, precision and recall metrics
    Y_pred, bag_clf = classf.Bagging_Classifier(X_train, Y_train, X_test,i)
    accuracies.append(round(metrics.accuracy_score(Y_test,Y_pred),2)*100)
    precisions.append(round(metrics.precision_score(Y_test,Y_pred),2)*100)
    recalls.append(round(metrics.recall_score(Y_test,Y_pred),2)*100)

BAG_scores_df['accuracies']=accuracies           #create new column in the dataframe with the accuracies
BAG_scores_df['precisions']=precisions       #create new column in the dataframe with the precisions
BAG_scores_df['recalls']=recalls         #create new column in the dataframe with the recalls
print(f"\nBagging ({dim_reduction} {n_features}) performance:\n")
print(BAG_scores_df)

#2. Estimators Visualisation
# visualise the performance metrics vs the number of nearest neighbors in a plot
fig, ax = plt.subplots()
ax.scatter(BAG_scores_df['k'], BAG_scores_df['accuracies'], c='b', label='Accuracy')
ax.scatter(BAG_scores_df['k'], BAG_scores_df['precisions'], c='g', label='Precision')
ax.scatter(BAG_scores_df['k'], BAG_scores_df['recalls'], c='r', label='Recall')
ax.set(ylabel='Performance (%)',xlabel='Number of Estimators K', ylim=[50, 100])
plt.legend(loc='lower right')
plt.grid(visible = True)
plt.title(f'Bagging ({dim_reduction} {n_features}) Performance vs Number of Estimators K', weight = 'bold')
plt.show()

# 3. Fit Bagging model with KNN for K = 13 and get accuracy score
K_bag = 14      #set the number of nearest neighbors for KNN for bagging
start_time = time.time()
Y_pred_BAG, bag_clf = classf.Bagging_Classifier(X_train, Y_train, X_test, K_bag)        #get predictions with best parameter
elapsed_time = time.time() - start_time
print(f"\nElapsed time to classify the data using Bagging ({dim_reduction} {n_features}) Classifier for K = {K_bag}: {elapsed_time:.2f} seconds")

# 4. Get Performance Scores
BAG_accuracy = round(accuracy_score(Y_test,Y_pred_BAG),2)*100     #get accuracy
BAG_precision = round(precision_score(Y_test,Y_pred_BAG),2)*100       #get precision
BAG_recall = round(recall_score(Y_test,Y_pred_BAG),2)*100
print(f'\nBagging ({dim_reduction} {n_features}) Accuracy Score on Test data: {BAG_accuracy}%')
print(f'\nBagging ({dim_reduction} {n_features}) Precision Score on Test data: {BAG_precision}%')
print(f'\nBagging ({dim_reduction} {n_features}) Recall Score on Test data: {BAG_recall}%')

# 5. Plot non-normalized confusion matrix
titles_options = [
    (f"Bagging ({dim_reduction} {n_features}) Confusion Matrix for K = {K_bag}", None),
    #(f"Bagging ({dim_reduction} {n_features}) Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        bag_clf,
        X_test,
        Y_test,
        display_labels=["No Tumor", "Tumor"],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
plt.show()
