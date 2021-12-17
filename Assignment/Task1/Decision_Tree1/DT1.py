# Date: 17/12/2021
# Institution: University College London
# Dept: EEE
# Class: ELEC0134
# SN: 18081659
# Description: Decision Tree Testing. See plots file for Results

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import sys
import time

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

# 3.Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(images_features,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}\n'.format(round(((len(Y_train)*1.0)/len(images_features)),3),round((len(Y_test)*1.0)/len(labels),3)))

# #Plot the features importances
# dt.plot_features_importances(X_train, Y_train, dim_reduction, n_features)

########################################## DT CLASSIFIER ######################################################

# 1. Fit Decision Tree model
tree_params={'criterion':'entropy'}     #define Tree parameters
start_time = time.time()        #start the time counter (to determine the time taken to classify the data
Y_pred_DT, dt_clf = classf.Decision_Tree_Classifier(X_train, Y_train, X_test, tree_params)      #classify the data using the Decision Tree Classifier
elapsed_time = time.time() - start_time     #get the elapsed time since the counter was started
print(f"Elapsed time to classify the data using Decision Tree ({dim_reduction} {n_features}) Classifier: {elapsed_time:.2f} seconds")

# 2. Get Performance Scores
DT_accuracy = round(accuracy_score(Y_test,Y_pred_DT),2)*100     #get accuracy
DT_precision = round(precision_score(Y_test,Y_pred_DT),2)*100       #get precision
DT_recall = round(recall_score(Y_test,Y_pred_DT),2)*100         #get recall
print(f'\nDecision Tree ({dim_reduction} {n_features}) Accuracy Score on Test data: {DT_accuracy}%')
print(f'\nDecision Tree ({dim_reduction} {n_features}) Precision Score on Test data: {DT_precision}%')
print(f'\nDecision Tree ({dim_reduction} {n_features}) Recall Score on Test data: {DT_recall}%')

# # 3. Decision Tree visualisation
# classf.visualise_tree(dt_clf)     #visualise Tree structure

# 4. Plot non-normalized confusion matrix
titles_options = [
    (f"Decision Tree ({dim_reduction} {n_features}) Confusion Matrix", None),
    #("Decision Tree ({dim_reduction}) Normalized confusion matrix", "true"),    #in case we want the normalised matrix
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
