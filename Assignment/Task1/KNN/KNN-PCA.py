
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
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
forest_importances, std = dt.get_features_importance_with_RF(np.array(X_train), np.array(Y_train))
fig, ax = plt.subplots()            #define the plot object
forest_importances.plot.bar(yerr=std, ax=ax)        #plot ar graph
ax.set_title("PCA Feature importances using MDI")       #set title
ax.set_ylabel("Mean decrease in impurity")      #set y-label
fig.tight_layout()
plt.show()

########################################## KNN WITH PCA CLASSIFIER ######################################################

#1. Estimators Tunning
KNN_scores_df = pd.DataFrame(list(range(1,16)), columns=["k"])
accuracies = []
precisions = []
recalls = []
estimators_range = [1, 16]
for i in range(estimators_range[0],estimators_range[1]):
    Y_pred, KNN_clf = dt.KNN_Classifier(X_train, Y_train, X_test,i)
    accuracies.append(round(metrics.accuracy_score(Y_test,Y_pred),2)*100)
    precisions.append(round(metrics.precision_score(Y_test,Y_pred),2)*100)
    recalls.append(round(metrics.recall_score(Y_test,Y_pred),2)*100)

KNN_scores_df['accuracies']=accuracies
KNN_scores_df['precisions']=precisions
KNN_scores_df['recalls']=recalls
print("\nKNN (PCA) performance:\n")
print(KNN_scores_df)

#2. Estimators Visualisation
fig, ax = plt.subplots()
ax.scatter(KNN_scores_df['k'], KNN_scores_df['accuracies'], c='b', label='Accuracy')
ax.scatter(KNN_scores_df['k'], KNN_scores_df['precisions'], c='g', label='Precision')
ax.scatter(KNN_scores_df['k'], KNN_scores_df['recalls'], c='r', label='Recall')
ax.set(title = 'KNN (PCA) Performance vs Number of Estimators K',
        ylabel='Performance (%)',xlabel='Number of Estimators K', ylim=[50, 110])
plt.legend(loc='lower right')
plt.grid(visible = True)
plt.title('KNN (PCA) Performance vs Number of Estimators K', weight = 'bold')
plt.show()

# 3. Fit KNN model for K = 6 and get accuracy score
start_time = time.time()
Y_pred_KNN, KNN_clf = dt.KNN_Classifier(X_train, Y_train, X_test, 6)
elapsed_time = time.time() - start_time
print(f"\nElapsed time to classify the data using KNN (PCA) Classifier for K = 6: {elapsed_time/60:.2f} minutes")

# 4. Get Performance Scores
KNN_accuracy = round(accuracy_score(Y_test,Y_pred_KNN),2)*100     #get accuracy
KNN_precision = round(precision_score(Y_test,Y_pred_KNN),2)*100       #get precision
KNN_recall = round(recall_score(Y_test,Y_pred_KNN),2)*100
print('\nKNN (PCA) Accuracy Score on Test data: {}%'.format(KNN_accuracy))
print('\nKNN (PCA) Precision Score on Test data: {}%'.format(KNN_precision))
print('\nKNN (PCA) Recall Score on Test data: {}%'.format(KNN_recall))

# 5. Plot non-normalized confusion matrix
titles_options = [
    ("KNN (PCA) Confusion matrix", None),
    #("KNN (PCA) Normalized confusion matrix", "true"),
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

