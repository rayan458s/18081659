
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

#Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(images_vectors,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}\n'.format(round(((len(Y_train)*1.0)/len(images_vectors)),3),round((len(Y_test)*1.0)/len(labels),3)))

########################################## BAGGING CLASSIFIER ######################################################

#1. Estimators Tunning
BAG_scores_df = pd.DataFrame(list(range(1,15)), columns=["k"])
accuracies = []
precisions = []
recalls = []
estimators_range = [1, 15]
for i in range(estimators_range[0],estimators_range[1]):
    Y_pred, bag_clf = dt.Bagging_Classifier(X_train, Y_train, X_test,i)
    accuracies.append(round(metrics.accuracy_score(Y_test,Y_pred),2)*100)
    precisions.append(round(metrics.precision_score(Y_test,Y_pred),2)*100)
    recalls.append(round(metrics.recall_score(Y_test,Y_pred),2)*100)


BAG_scores_df['accuracies']=accuracies
BAG_scores_df['precisions']=precisions
BAG_scores_df['recalls']=recalls
print("\nBagging (No Feature Selection) performance:\n")
print(BAG_scores_df)

#2. Estimators Visualisation
fig, ax = plt.subplots()
ax.scatter(BAG_scores_df['k'], BAG_scores_df['accuracies'], c='b', label='Accuracy')
ax.scatter(BAG_scores_df['k'], BAG_scores_df['precisions'], c='g', label='Precision')
ax.scatter(BAG_scores_df['k'], BAG_scores_df['recalls'], c='r', label='Recall')
ax.set(title = 'Bagging (No Features Selection) Accuracy vs Number of Estimators (K)',
        ylabel='Accuracy',xlabel='Number of Estimators K', ylim=[50, 110])
plt.legend(loc='lower right')
plt.grid(visible = True)
plt.title('Bagging (No Features Selection) Accuracy vs Number of Estimators (K)', weight = 'bold')
plt.show()

# 3. Fit Bagging model with KNN for K = 10 and get accuracy score
start_time = time.time()
Y_pred_BAG, bag_clf = dt.Bagging_Classifier(X_train, Y_train, X_test, 10)
elapsed_time = time.time() - start_time
print(f"\nElapsed time to classify the data using Bagging (No Features Selection) Classifier for K = 10: {elapsed_time/60:.2f} minutes")

# 4. Get Performance Scores
BAG_accuracy = round(accuracy_score(Y_test,Y_pred_BAG),2)*100     #get accuracy
BAG_precision = round(precision_score(Y_test,Y_pred_BAG),2)*100       #get precision
BAG_recall = round(recall_score(Y_test,Y_pred_BAG),2)*100
print('\nBagging (No Features Selection) Accuracy Score on Test data: {}%'.format(BAG_accuracy))
print('\nBagging (No Features Selection) Precision Score on Test data: {}%'.format(BAG_precision))
print('\nBagging (No Features Selection) Recall Score on Test data: {}%'.format(BAG_recall))

# 5. Plot non-normalized confusion matrix
titles_options = [
    ("Bagging (No Features Selection) Confusion matrix", None),
    #("Bagging (No Features Selection) Normalized confusion matrix", "true"),
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
BOOST_scores_df = pd.DataFrame(list(range(1,15)), columns=["k"])
accuracies = []
precisions = []
recalls = []
estimators_range = [1, 15]
for i in range(estimators_range[0],estimators_range[1]):
    Y_pred, boost_clf = dt.Boosting_Classifier(X_train, Y_train, X_test,i)
    accuracies.append(round(metrics.accuracy_score(Y_test,Y_pred),2)*100)
    precisions.append(round(metrics.precision_score(Y_test,Y_pred),2)*100)
    recalls.append(round(metrics.recall_score(Y_test,Y_pred),2)*100)
    print('Scores for k={} computed'.format(i))

BOOST_scores_df['accuracies']=accuracies
BOOST_scores_df['precisions']=precisions
BOOST_scores_df['recalls']=recalls
print("\nBoosting (No Feature Selection) performance:\n")
print(BOOST_scores_df)

#2. Estimators Visualisation
fig, ax = plt.subplots()
ax.scatter(BOOST_scores_df['k'], BOOST_scores_df['accuracies'], c='b', label='Accuracy')
ax.scatter(BOOST_scores_df['k'], BOOST_scores_df['precisions'], c='g', label='Precision')
ax.scatter(BOOST_scores_df['k'], BOOST_scores_df['recalls'], c='r', label='Recall')
ax.set(title = 'Boosting (No Features Selection) Accuracy vs Number of Estimators (K)',
        ylabel='Accuracy',xlabel='Number of Estimators K', ylim=[50, 100])
plt.legend(loc='lower right')
plt.grid(visible = True)
plt.title('Boosting (No Features Selection) Accuracy vs Number of Estimators (K)', weight = 'bold')
plt.show()

# 3. Fit ADABOOST model with Decision Three for K = 7
start_time = time.time()
Y_pred_BOOST, boost_clf = dt.Boosting_Classifier(X_train, Y_train, X_test, 7)
elapsed_time = time.time() - start_time
print(f"\nElapsed time to classify the data using Boosting (No Features Selection) Classifier for K = 7: {elapsed_time/60:.2f} minutes")

# 4. Get Performance Scores
BOOST_accuracy = round(accuracy_score(Y_test,Y_pred_BOOST),2)*100     #get accuracy
BOOST_precision = round(precision_score(Y_test,Y_pred_BOOST),2)*100       #get precision
BOOST_recall = round(recall_score(Y_test,Y_pred_BOOST),2)*100
print('\nBoosting (No Features Selection) Accuracy Score on Test data: {}%'.format(BOOST_accuracy))
print('\nBoosting (No Features Selection) Precision Score on Test data: {}%'.format(BOOST_precision))
print('\nBoosting (No Features Selection) Recall Score on Test data: {}%'.format(BOOST_recall))

# 5. Plot non-normalized confusion matrix
titles_options = [
    ("Boosting (No Features Selection) Confusion matrix", None),
    #("Boosting (No Features Selection) Normalized confusion matrix", "true"),
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