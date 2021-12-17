# Date: 17/12/2021
# Institution: University College London
# Dept: EEE
# Class: ELEC0134
# SN: 18081659
# Description: DT with Boosting Testing. See plots file for results.

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
import sys
import matplotlib.pyplot as plt
import time
import pandas as pd

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


########################################## BOOSTING ######################################################

#1. Estimators Tunning for Decision Tree Estimator
BOOST_scores_df = pd.DataFrame(list(range(1,20)), columns=["k"])
accuracies = []
precisions = []
recalls = []
estimators_range = [1, 20]
for i in range(estimators_range[0],estimators_range[1]):        #get scores for the entire estimators range
    Y_pred, boost_clf = classf.Boosting_Classifier(X_train, Y_train, X_test,None,i)
    accuracies.append(round(metrics.accuracy_score(Y_test,Y_pred),2)*100)
    precisions.append(round(metrics.precision_score(Y_test,Y_pred),2)*100)
    recalls.append(round(metrics.recall_score(Y_test,Y_pred),2)*100)

BOOST_scores_df['accuracies'] = accuracies
BOOST_scores_df['precisions'] = precisions
BOOST_scores_df['recalls'] = recalls
print(f"\nBoosting with DT ({dim_reduction} {n_features}) performance:\n")
print(BOOST_scores_df)

#2. Estimators Visualisation for Decision Tree Estimators
fig, ax = plt.subplots()
ax.scatter(BOOST_scores_df['k'], BOOST_scores_df['accuracies'], c='b', label='Accuracy')
ax.scatter(BOOST_scores_df['k'], BOOST_scores_df['precisions'], c='g', label='Precision')
ax.scatter(BOOST_scores_df['k'], BOOST_scores_df['recalls'], c='r', label='Recall')
ax.set(ylabel='Performance (%)',xlabel='Number of Estimators K', ylim=[50, 110])
plt.legend(loc='lower right')
plt.grid(visible = True)
plt.title(f'Boosting with DT ({dim_reduction} {n_features}) Performance vs Number of Estimators K', weight = 'bold')
plt.show()

# 5. Fit ADABOOST model with Decision Three with best hyperparameter
K_boost = 7
start_time = time.time()
Y_pred_BOOST, boost_clf = classf.Boosting_Classifier(X_train, Y_train, X_test, None, K_boost)   #get prediction with best hyperparameter
elapsed_time = time.time() - start_time
print(f"\nElapsed time to classify the data using Boosting with Decision Three ({dim_reduction} {n_features}) Classifier for K = {K_boost}: {elapsed_time:.2f} seconds")

# 6. Get Performance Scores
BOOST_accuracy = round(accuracy_score(Y_test,Y_pred_BOOST),2)*100     #get accuracy
BOOST_precision = round(precision_score(Y_test,Y_pred_BOOST),2)*100       #get precision
BOOST_recall = round(recall_score(Y_test,Y_pred_BOOST),2)*100       #get recall
print(f'\nBoosting with Decision Three ({dim_reduction} {n_features}) Accuracy Score on Test data: {BOOST_accuracy}%')
print(f'\nBoosting with Decision Three ({dim_reduction} {n_features}) Precision Score on Test data: {BOOST_precision}%')
print(f'\nBoosting with Decision Three ({dim_reduction} {n_features}) Recall Score on Test data: {BOOST_recall}%')

# 7. Plot non-normalized confusion matrix
titles_options = [
    (f"Boosting with Decision Three ({dim_reduction} {n_features}) Confusion Matrix for n_est = {K_boost}", None),
    #(f"Boosting ({dim_reduction} {n_features}) Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        boost_clf,
        X_test,
        Y_test,
        display_labels=["No Tumor", "Tumor"],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
plt.show()
