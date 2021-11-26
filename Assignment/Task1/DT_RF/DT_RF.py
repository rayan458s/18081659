
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import matplotlib.pyplot as plt
import time

from Assignment.Package import data_processing as dt

########################################## DATA PROCESSING ######################################################

images_vectors, labels = dt.process_data()

#Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(images_vectors,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}\n'.format(round(((len(Y_train)*1.0)/len(images_vectors)),3),round((len(Y_test)*1.0)/len(labels),3)))


########################################## DT CLASSIFIER ######################################################

# 1. Fit Decision Three model
tree_params={'criterion':'entropy'}
start_time = time.time()
Y_pred_DT, dt_clf = dt.Decision_Tree_Classifier(X_train, Y_train, X_test, tree_params)
elapsed_time = time.time() - start_time
print(f"Elapsed time to classify the data using Decision Three (No Features Selection) Classifier: {elapsed_time/60:.2f} minutes")

# 2. Get Performance Scores
DT_accuracy = round(accuracy_score(Y_test,Y_pred_DT),2)*100     #get accuracy
DT_precision = round(precision_score(Y_test,Y_pred_DT),2)*100       #get precision
DT_recall = round(recall_score(Y_test,Y_pred_DT),2)*100
print('\nDecision Tree (No Features Selection) Accuracy Score on Test data: {}%'.format(DT_accuracy))
print('\nDecision Tree (No Features Selection) Precision Score on Test data: {}%'.format(DT_precision))
print('\nDecision Tree (No Features Selection) Recall Score on Test data: {}%'.format(DT_recall))

# 3. Plot non-normalized confusion matrix
titles_options = [
    ("Decision Three (No Features Selection) Confusion Matrix", None),
    #("Regular Decision Three Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        dt_clf,
        X_test,
        Y_test,
        display_labels=["No Tumor", "Tumor"],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
plt.show()

# 4. Hyperparameter Tuning
tree_params = {'criterion': 'entropy', 'min_samples_split':50}
start_time = time.time()
Y_pred_DT2, dt_clf_2 = dt.Decision_Tree_Classifier(X_train, Y_train, X_test, tree_params)
elapsed_time = time.time() - start_time
print(f"\nElapsed time to classify the data using Decision Three (No Features Selection) Classifier after hyperparameters tuning: {elapsed_time/60:.2f} minutes")

# 2. Get Performance Scores
DT2_accuracy = round(accuracy_score(Y_test,Y_pred_DT2),2)*100     #get accuracy
DT2_precision = round(precision_score(Y_test,Y_pred_DT2),2)*100       #get precision
DT2_recall = round(recall_score(Y_test,Y_pred_DT2),2)*100
print('\nDecision Tree (No Features Selection) (after tuning) Accuracy Score on Test data: {}%'.format(DT2_accuracy))
print('\nDecision Tree (No Features Selection) (after tuning) Precision Score on Test data: {}%'.format(DT2_precision))
print('\nDecision Tree (No Features Selection) (after tuning) Recall Score on Test data: {}%'.format(DT2_recall))


# 6. Plot non-normalized confusion matrix after tuning
titles_options = [
    ("Decision Three (No Features Selection) Confusion Matrix After Tuning", None),
    #("Regular Decision Three Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        dt_clf,
        X_test,
        Y_test,
        display_labels=["No Tumor", "Tumor"],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
plt.show()

########################################## RF CLASSIFIER ######################################################

# 1. Fit Random Forest model and get accuracy score
start_time = time.time()
Y_pred_RF, rf_clf = dt.Random_Forest_Classifier(X_train, Y_train, X_test)
elapsed_time = time.time() - start_time
print(f"\nElapsed time to classify the data using Random Forest (No Features Selection) Classifier: {elapsed_time/60:.2f} minutes")

# 2. Get Performance Scores
RF_accuracy = round(accuracy_score(Y_test,Y_pred_RF),2)*100     #get accuracy
RF_precision = round(precision_score(Y_test,Y_pred_RF),2)*100       #get precision
RF_recall = round(recall_score(Y_test,Y_pred_RF),2)*100
print('\nRandom Forest (No Features Selection) Accuracy Score on Test data: {}%'.format(RF_accuracy))
print('\nRandom Forest (No Features Selection) Precision Score on Test data: {}%'.format(RF_accuracy))
print('\nRandom Forest (No Features Selection) Recall Score on Test data: {}%'.format(RF_accuracy))

# 3. Plot non-normalized confusion matrix
titles_options = [
    ("Random Forest (No Features Selection) Confusion Matrix", None),
    #("Regular Random Forest Normalized confusion matrix", "true"),
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
