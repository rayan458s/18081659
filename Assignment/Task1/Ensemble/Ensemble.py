
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

from Assignment.Package import data_processing as dt

########################################## DATA PROCESSING ######################################################

images_vectors, labels = dt.process_data()      #process all the images into pixel vectors and get the labels
#WARNING ONLY 5 AND 10 FEATURES MODES HAVE BEEN PROCESSED FOR ANOVA and PCA, IF YOU WISH TO USE A DIFFERENT NUMBER OF
#FEATURES YOU NEED TO FIRST USE THE dt.process_ANOVA_features(n_features) or dt.process_PCA_features(n_features) functions
n_features = 10     #define the number of features to select/project from the pixels vectors
dim_reduction = "PCA"
if dim_reduction == "ANOVA":
    images_features = dt.get_ANOVA_features(n_features)     #select the desired number of feartures using ANOVA
elif dim_reduction == "PCA":
    images_features = dt.get_PCA_features(n_features)     #project the desired number of feartures using PCA
else:
    print('\nNot a valid dimensionality reduction technique\n')


#Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(images_features,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}\n'.format(round(((len(Y_train)*1.0)/len(images_features)),3),round((len(Y_test)*1.0)/len(labels),3)))

# #Plot the features importances
# forest_importances, std = dt.get_features_importance_with_RF(X_train, Y_train)
# fig, ax = plt.subplots()            #define the plot object
# forest_importances.plot.bar(yerr=std, ax=ax)        #plot bar graph
# ax.set_title(f"{dim_reduction} ({n_features}) Feature Importances Using MDI")       #set title
# ax.set_ylabel("Mean decrease in impurity")      #set y-label
# fig.tight_layout()
# plt.show()

########################################## BAGGING CLASSIFIER ######################################################

#1. Estimators Tunning
BAG_scores_df = pd.DataFrame(list(range(1,16)), columns=["k"])  #create pandas dataframe to containe the performance metrics for difference values of K
accuracies = []
precisions = []
recalls = []
estimators_range = [1, 16]      #the range to test the number of nearest neighbors
for i in range(estimators_range[0],estimators_range[1]):    #classify the data for every value of K using bagging and get accuracy, precision and recall metrics
    Y_pred, bag_clf = dt.Bagging_Classifier(X_train, Y_train, X_test,i)
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
K_bag = 13      #set the number of nearest neighbors for KNN for bagging
start_time = time.time()
Y_pred_BAG, bag_clf = dt.Bagging_Classifier(X_train, Y_train, X_test, K_bag)
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

########################################## BOOSTING CLASSIFIER ######################################################

#1. Estimators Tunning for Decision Three Estimator
BOOST_scores_df = pd.DataFrame(list(range(1,16)), columns=["k"])
accuracies = []
precisions = []
recalls = []
estimators_range = [1, 16]
for i in range(estimators_range[0],estimators_range[1]):
    Y_pred, boost_clf = dt.Boosting_Classifier(X_train, Y_train, X_test,None,i)
    accuracies.append(round(metrics.accuracy_score(Y_test,Y_pred),2)*100)
    precisions.append(round(metrics.precision_score(Y_test,Y_pred),2)*100)
    recalls.append(round(metrics.recall_score(Y_test,Y_pred),2)*100)

BOOST_scores_df['accuracies'] = accuracies
BOOST_scores_df['precisions'] = precisions
BOOST_scores_df['recalls'] = recalls
print(f"\nBoosting with DT ({dim_reduction} {n_features}) performance:\n")
print(BOOST_scores_df)

#2. Estimators Visualisation for Decision Three Estimator
fig, ax = plt.subplots()
ax.scatter(BOOST_scores_df['k'], BOOST_scores_df['accuracies'], c='b', label='Accuracy')
ax.scatter(BOOST_scores_df['k'], BOOST_scores_df['precisions'], c='g', label='Precision')
ax.scatter(BOOST_scores_df['k'], BOOST_scores_df['recalls'], c='r', label='Recall')
ax.set(ylabel='Performance (%)',xlabel='Number of Estimators K', ylim=[50, 110])
plt.legend(loc='lower right')
plt.grid(visible = True)
plt.title(f'Boosting with DT ({dim_reduction} {n_features}) Performance vs Number of Estimators K', weight = 'bold')
plt.show()

#3. Estimators Tunning for Random Forest Estimator
BOOST_scores_df = pd.DataFrame(list(range(1,16)), columns=["k"])
accuracies = []
precisions = []
recalls = []
estimators_range = [1, 16]
for i in range(estimators_range[0],estimators_range[1]):
    Y_pred, boost_clf = dt.Boosting_Classifier(X_train, Y_train, X_test,RandomForestClassifier(n_estimators=100),i)
    accuracies.append(round(metrics.accuracy_score(Y_test,Y_pred),2)*100)
    precisions.append(round(metrics.precision_score(Y_test,Y_pred),2)*100)
    recalls.append(round(metrics.recall_score(Y_test,Y_pred),2)*100)

BOOST_scores_df['accuracies'] = accuracies
BOOST_scores_df['precisions'] = precisions
BOOST_scores_df['recalls'] = recalls
print(f"\nBoosting with Random Forest ({dim_reduction} {n_features}) performance:\n")
print(BOOST_scores_df)

#4. Estimators Visualisation for Random Forest Estimator
fig, ax = plt.subplots()
ax.scatter(BOOST_scores_df['k'], BOOST_scores_df['accuracies'], c='b', label='Accuracy')
ax.scatter(BOOST_scores_df['k'], BOOST_scores_df['precisions'], c='g', label='Precision')
ax.scatter(BOOST_scores_df['k'], BOOST_scores_df['recalls'], c='r', label='Recall')
ax.set(ylabel='Performance (%)',xlabel='Number of Estimators K', ylim=[50, 110])
plt.legend(loc='lower right')
plt.grid(visible = True)
plt.title(f'Boosting with RF ({dim_reduction} {n_features}) Performance vs Number of Estimators K', weight = 'bold')
plt.show()

# 5. Fit ADABOOST model with Decision Three for K = 2
K_boost = 2
start_time = time.time()
Y_pred_BOOST, boost_clf = dt.Boosting_Classifier(X_train, Y_train, X_test, None, K_boost)
elapsed_time = time.time() - start_time
print(f"\nElapsed time to classify the data using Boosting with Decision Three ({dim_reduction} {n_features}) Classifier for K = {K_boost}: {elapsed_time:.2f} seconds")

# 6. Get Performance Scores
BOOST_accuracy = round(accuracy_score(Y_test,Y_pred_BOOST),2)*100     #get accuracy
BOOST_precision = round(precision_score(Y_test,Y_pred_BOOST),2)*100       #get precision
BOOST_recall = round(recall_score(Y_test,Y_pred_BOOST),2)*100
print(f'\nBoosting with Decision Three ({dim_reduction} {n_features}) Accuracy Score on Test data: {BOOST_accuracy}%')
print(f'\nBoosting with Decision Three ({dim_reduction} {n_features}) Precision Score on Test data: {BOOST_precision}%')
print(f'\nBoosting with Decision Three ({dim_reduction} {n_features}) Recall Score on Test data: {BOOST_recall}%')

# 7. Plot non-normalized confusion matrix
titles_options = [
    (f"Boosting with Decision Three ({dim_reduction} {n_features}) Confusion Matrix for K = {K_boost}", None),
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