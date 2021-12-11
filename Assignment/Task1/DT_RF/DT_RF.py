
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import matplotlib.pyplot as plt
import time

from Assignment.Functions import data_processing_T1 as dt
from Assignment.Functions import Classifiers as classf

########################################## DATA PROCESSING ######################################################

images_vectors, labels = dt.process_data()      #process all the images into pixel vectors and get the labels

#WARNING ONLY 5 AND 10 FEATURES MODES HAVE BEEN PROCESSED FOR ANOVA and PCA, IF YOU WISH TO USE A DIFFERENT NUMBER OF
#FEATURES YOU NEED TO FIRST USE THE dt.process_ANOVA_features(n_features) or dt.process_PCA_features(n_features) functions

n_features = 5     #define the number of features to select/project from the pixels vectors
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
# dt.plot_features_importances(X_train, Y_train, dim_reduction, n_features)

########################################## DT CLASSIFIER ######################################################

# 1. Fit Decision Three model
tree_params={'criterion':'entropy'}     #define three parameters
start_time = time.time()        #start the time counter (to determine the time taken to classify the data
Y_pred_DT, dt_clf = classf.Decision_Tree_Classifier(X_train, Y_train, X_test, tree_params)      #classify the data using the Decision Three Classifier
elapsed_time = time.time() - start_time     #get the elapsed time since the counter was started
print(f"Elapsed time to classify the data using Decision Three ({dim_reduction} {n_features}) Classifier: {elapsed_time:.2f} seconds")

# 2. Get Performance Scores
DT_accuracy = round(accuracy_score(Y_test,Y_pred_DT),2)*100     #get accuracy
DT_precision = round(precision_score(Y_test,Y_pred_DT),2)*100       #get precision
DT_recall = round(recall_score(Y_test,Y_pred_DT),2)*100         #get recall
print(f'\nDecision Tree ({dim_reduction} {n_features}) Accuracy Score on Test data: {DT_accuracy}%')
print(f'\nDecision Tree ({dim_reduction} {n_features}) Precision Score on Test data: {DT_precision}%')
print(f'\nDecision Tree ({dim_reduction} {n_features}) Recall Score on Test data: {DT_recall}%')

# # 3. Decision Three visualisation
# classf.visualise_tree(dt_clf)     #visualise three structure

# 4. Plot non-normalized confusion matrix
titles_options = [
    (f"Decision Three ({dim_reduction} {n_features}) Confusion Matrix", None),
    #("Decision Three ({dim_reduction}) Normalized confusion matrix", "true"),    #in case we want the normalised matrix
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

# 5. Hyperparameter Tuning
tree_params = {'criterion': 'entropy', 'min_samples_split':50}      #change the min number of sample (mini-batch size)
start_time = time.time()       #get the elapsed time since the counter was started
Y_pred_DT2, dt_clf_2 = classf.Decision_Tree_Classifier(X_train, Y_train, X_test, tree_params)   #classify the data using the Decision Three Classifier with new parameters
elapsed_time = time.time() - start_time         #get the elapsed time since the counter was started
print(f"\nElapsed time to classify the data using Decision Three ({dim_reduction}{n_features}) Classifier after hyperparameters tuning: {elapsed_time:.2f} seconds")

# 6. Get Performance Scores
DT2_accuracy = round(accuracy_score(Y_test,Y_pred_DT2),2)*100     #get accuracy
DT2_precision = round(precision_score(Y_test,Y_pred_DT2),2)*100       #get precision
DT2_recall = round(recall_score(Y_test,Y_pred_DT2),2)*100               #get recall
print(f'\nDecision Tree ({dim_reduction} {n_features}) Accuracy Score on Test data after tuning: {DT2_accuracy}%')
print(f'\nDecision Tree ({dim_reduction} {n_features}) Precision Score on Test data after tuning: {DT2_precision}%')
print(f'\nDecision Tree ({dim_reduction} {n_features}) Recall Score on Test data after tuning: {DT2_recall}%')


# 7. Plot non-normalized confusion matrix after tuning
titles_options = [
    (f"Decision Three ({dim_reduction} {n_features}) Confusion Matrix After Tuning", None),
    #("Decision Three ({dim_reduction}) Normalized confusion matrix", "true"),    # in case we want the normalised matrix
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
start_time = time.time()        #get the elapsed time since the counter was started
Y_pred_RF, rf_clf = classf.Random_Forest_Classifier(X_train, Y_train, X_test)           #classify the data using the Random Forest Classifier
elapsed_time = time.time() - start_time         #get the elapsed time since the counter was started
print(f"\nElapsed time to classify the data using Random Forest ({dim_reduction}{n_features}) Classifier: {elapsed_time:.2f} seconds")

# 2. Get Performance Scores
RF_accuracy = round(accuracy_score(Y_test,Y_pred_RF),2)*100     #get accuracy
RF_precision = round(precision_score(Y_test,Y_pred_RF),2)*100       #get precision
RF_recall = round(recall_score(Y_test,Y_pred_RF),2)*100
print(f'\nRandom Forest ({dim_reduction} {n_features}) Accuracy Score on Test data: {RF_accuracy}%')
print(f'\nRandom Forest ({dim_reduction} {n_features}) Precision Score on Test data: {RF_precision}%')
print(f'\nRandom Forest ({dim_reduction} {n_features}) Recall Score on Test data: {RF_recall}%')

# # 3. Random Forest  visualisation
# for index in range(0, 5):
#     classf.visualise_tree(rf_clf.estimators_[index])      #visualise 5 of the  threes structures of the random forest

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
