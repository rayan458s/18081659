# Date: 17/12/2021
# Institution: University College London
# Dept: EEE
# Class: ELEC0134
# SN: 18081659
# Description: Random Forest Testing. See plots file for Results

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
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

# #Plot the features importances
# dt.plot_features_importances(X_train, Y_train, dim_reduction, n_features)

########################################## RF CLASSIFIER ######################################################

#1. Number of Trees Tunning
#print(rf_clf.get_params())
RF_scores_df = pd.DataFrame(list(range(10,320,20)), columns=["T"])      #create pandas dataframe to containe the performance metrics for difference values of K
accuracies = []
precisions = []
recalls = []
estimators_range = [10,320]      #the range to test the number of nearest neighbors
for i in range(estimators_range[0],estimators_range[1],20):        #classify the data for every value of K and get accuracy, precision and recall metrics
    Y_pred_RF, rf_clf = classf.Random_Forest_Classifier(X_train, Y_train, X_test,i)
    accuracies.append(round(accuracy_score(Y_test,Y_pred_RF),2)*100)
    precisions.append(round(precision_score(Y_test,Y_pred_RF),2)*100)
    recalls.append(round(recall_score(Y_test,Y_pred_RF),2)*100)

RF_scores_df['accuracies']=accuracies      #create new column in the dataframe with the accuracies
RF_scores_df['precisions']=precisions         #create new column in the dataframe with the precisions
RF_scores_df['recalls']=recalls                    #create new column in the dataframe with the recalls
print(f"\nRF ({dim_reduction} {n_features}) performance:\n")
print(RF_scores_df)

#2. Performance Vs Number of Trees Visualisation
# visualise the performance metrics vs the number of trees in a plot
fig, ax = plt.subplots()
ax.scatter(RF_scores_df['T'], RF_scores_df['accuracies'], c='b', label='Accuracy')
ax.scatter(RF_scores_df['T'], RF_scores_df['precisions'], c='g', label='Precision')
ax.scatter(RF_scores_df['T'], RF_scores_df['recalls'], c='r', label='Recall')
ax.set(ylabel='Performance (%)',xlabel='Number of Trees', ylim=[50, 110])
plt.legend(loc='lower right')
plt.grid(visible = True)
plt.title(f'Random Forest ({dim_reduction} {n_features}) Performance vs Number of Trees', weight = 'bold')
plt.show()

# 3. Classify using Random Forest with best hyperparameters
n_trees = 30
start_time = time.time()        #start the counter
Y_pred_RF, rf_clf = classf.Random_Forest_Classifier(X_train, Y_train, X_test,n_trees)
elapsed_time = time.time() - start_time         #get the elapsed time since the counter was started
print(f"\nElapsed time to classify the data with Random Forest ({dim_reduction} {n_features}) Classifier: {elapsed_time:.2f} seconds")

# 4. Get Performance Scores
RF_accuracy = round(accuracy_score(Y_test,Y_pred_RF),2)*100     #get accuracy
RF_precision = round(precision_score(Y_test,Y_pred_RF),2)*100       #get precision
RF_recall = round(recall_score(Y_test,Y_pred_RF),2)*100               #get recall
print(f'\nRF ({dim_reduction} {n_features}) Accuracy Score on Test data: {RF_accuracy}%')
print(f'\nRF ({dim_reduction} {n_features}) Precision Score on Test data: {RF_precision}%')
print(f'\nRF ({dim_reduction} {n_features}) Recall Score on Test data: {RF_recall}%')


# 4. Plot non-normalized confusion matrix
titles_options = [
    (f"Random Forest ({dim_reduction} {n_features}) Confusion Matrix", None),
    #("{dim_reduction} Random Forest Normalized confusion matrix", "true"),       #In case we want the normalised matrix
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        rf_clf,
        X_test,
        Y_test,
        display_labels=["No Tumor", "Tumor"],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
plt.show()


# # 5. Random Forest  visualisation
# for index in range(0, 5):
#    classf.visualise_tree(rf_clf.estimators_[index])      #visualise 5 of the  Trees structures of the random forest
