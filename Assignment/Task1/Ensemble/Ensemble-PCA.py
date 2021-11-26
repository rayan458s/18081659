
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

from Assignment.Package import data_processing as dt

########################################## DATA PROCESSING ######################################################

images_vectors, labels = dt.process_data()
images_features = dt.process_PCA_features(10)

#Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(images_features,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}\n'.format(round(((len(Y_train)*1.0)/len(images_features)),3),round((len(Y_test)*1.0)/len(labels),3)))

#Plot the features importances
forest_importances, std = dt.get_features_importance_with_RF(X_train, Y_train)
fig, ax = plt.subplots()            #define the plot object
forest_importances.plot.bar(yerr=std, ax=ax)        #plot ar graph
ax.set_title("PCA Feature importances using MDI")       #set title
ax.set_ylabel("Mean decrease in impurity")      #set y-label
fig.tight_layout()
plt.show()

########################################## BAGGING CLASSIFIER ######################################################

#1. Estimators Tunning
BAG_scores_df = pd.DataFrame(list(range(1,16)), columns=["k"])
accuracies = []
precisions = []
recalls = []
estimators_range = [1, 16]
for i in range(estimators_range[0],estimators_range[1]):
    Y_pred, bag_clf = dt.Bagging_Classifier(X_train, Y_train, X_test,i)
    accuracies.append(round(metrics.accuracy_score(Y_test,Y_pred),2)*100)
    precisions.append(round(metrics.precision_score(Y_test,Y_pred),2)*100)
    recalls.append(round(metrics.recall_score(Y_test,Y_pred),2)*100)

BAG_scores_df['accuracies']=accuracies
BAG_scores_df['precisions']=precisions
BAG_scores_df['recalls']=recalls
print("\nBagging (PCA) performance:\n")
print(BAG_scores_df)

#2. Estimators Visualisation
fig, ax = plt.subplots()
ax.scatter(BAG_scores_df['k'], BAG_scores_df['accuracies'], c='b', label='Accuracy')
ax.scatter(BAG_scores_df['k'], BAG_scores_df['precisions'], c='g', label='Precision')
ax.scatter(BAG_scores_df['k'], BAG_scores_df['recalls'], c='r', label='Recall')
ax.set(title = 'Bagging (PCA) Performance vs Number of Estimators K',
        ylabel='Performance (%)',xlabel='Number of Estimators K', ylim=[50, 110])
plt.legend(loc='lower right')
plt.grid(visible = True)
plt.title('Bagging (PCA) Performance vs Number of Estimators K', weight = 'bold')
plt.show()

# 3. Fit Bagging model with KNN for K = 10 and get accuracy score
start_time = time.time()
Y_pred_BAG, bag_clf = dt.Bagging_Classifier(X_train, Y_train, X_test, 10)
elapsed_time = time.time() - start_time
print(f"\nElapsed time to classify the data using Bagging (PCA) Classifier for K = 10: {elapsed_time/60:.2f} minutes")

# 4. Get Performance Scores
BAG_accuracy = round(accuracy_score(Y_test,Y_pred_BAG),2)*100     #get accuracy
BAG_precision = round(precision_score(Y_test,Y_pred_BAG),2)*100       #get precision
BAG_recall = round(recall_score(Y_test,Y_pred_BAG),2)*100
print('\nBagging (PCA) Accuracy Score on Test data: {}%'.format(BAG_accuracy))
print('\nBagging (PCA) Precision Score on Test data: {}%'.format(BAG_precision))
print('\nBagging (PCA) Recall Score on Test data: {}%'.format(BAG_recall))

# 5. Plot non-normalized confusion matrix
titles_options = [
    ("Bagging (PCA) Confusion matrix", None),
    #("Bagging (PCA) Normalized confusion matrix", "true"),
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

#1. Estimators Tunning
BOOST_scores_df = pd.DataFrame(list(range(1,16)), columns=["k"])
accuracies = []
precisions = []
recalls = []
estimators_range = [1, 16]
for i in range(estimators_range[0],estimators_range[1]):
    Y_pred, boost_clf = dt.Boosting_Classifier(X_train, Y_train, X_test,i)
    accuracies.append(round(metrics.accuracy_score(Y_test,Y_pred),2)*100)
    precisions.append(round(metrics.precision_score(Y_test,Y_pred),2)*100)
    recalls.append(round(metrics.recall_score(Y_test,Y_pred),2)*100)

BOOST_scores_df['accuracies'] = accuracies
BOOST_scores_df['precisions'] = precisions
BOOST_scores_df['recalls'] = recalls
print("\nBoosting (PCA) performance:\n")
print(BOOST_scores_df)

#2. Estimators Visualisation
fig, ax = plt.subplots()
ax.scatter(BOOST_scores_df['k'], BOOST_scores_df['accuracies'], c='b', label='Accuracy')
ax.scatter(BOOST_scores_df['k'], BOOST_scores_df['precisions'], c='g', label='Precision')
ax.scatter(BOOST_scores_df['k'], BOOST_scores_df['recalls'], c='r', label='Recall')
ax.set(title = 'Boosting (PCA) Performance vs Number of Estimators K',
        ylabel='Performance (%)',xlabel='Number of Estimators K', ylim=[50, 110])
plt.legend(loc='lower right')
plt.grid(visible = True)
plt.title('Boosting (PCA) Performance vs Number of Estimators K', weight = 'bold')
plt.show()

# 3. Fit ADABOOST model with Decision Three for K = 4
start_time = time.time()
Y_pred_BOOST, boost_clf = dt.Boosting_Classifier(X_train, Y_train, X_test, 4)
elapsed_time = time.time() - start_time
print(f"\nElapsed time to classify the data using Boosting (PCA) Classifier for K =4: {elapsed_time/60:.2f} minutes")

# 4. Get Performance Scores
BOOST_accuracy = round(accuracy_score(Y_test,Y_pred_BOOST),2)*100     #get accuracy
BOOST_precision = round(precision_score(Y_test,Y_pred_BOOST),2)*100       #get precision
BOOST_recall = round(recall_score(Y_test,Y_pred_BOOST),2)*100
print('\nBoosting (PCA) Accuracy Score on Test data: {}%'.format(BOOST_accuracy))
print('\nBoosting (PCA) Precision Score on Test data: {}%'.format(BOOST_precision))
print('\nBoosting (PCA) Recall Score on Test data: {}%'.format(BOOST_recall))

# 5. Plot non-normalized confusion matrix
titles_options = [
    ("Boosting (PCA) Confusion matrix", None),
    #("PCA Boosting Normalized confusion matrix", "true"),
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