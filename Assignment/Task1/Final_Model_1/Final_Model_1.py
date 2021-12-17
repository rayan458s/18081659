# Date: 17/12/2021
# Institution: University College London
# Dept: EEE
# Class: ELEC0134
# SN: 18081659
# Description: Final Task 1 model implementation. See plots file for Test Results

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.model_selection import ShuffleSplit

from Assignment.Functions import data_processing as dt


# 1. Get the input data (matrix of image vectors and output data (vector of labels)
print('\nLoading Initial Dataset.')
images_vectors, labels = dt.process_data()      #process all the images into pixel vectors and get the labels

# 2. Transform inputs into feature vectors
print('\nGetting Features.\n')
dim_reduction = 'PCA'       #define the DR method
n_features = 10     #define the number of features to select/project from the pixels vectors
filepath = '/Users/rayan/PycharmProjects/AMLS/Assignment/Task1/Features/'+dim_reduction+'-'+str(n_features)+'.dat'  #get corresponding filepath
images_features = dt.get_features(filepath)     #project the desired number of feartures using PCA
X, Y = images_features, labels

# #Plot the features importances
# dt.plot_features_importances(X, Y, dim_reduction, n_features)#

# 3. Define the initial Random Forest Classifier Object
rf_clf = RandomForestClassifier(n_estimators=30)    #create Random Forest claassifier instance

# 4. Cross Validation
print('\nCross Validation Stage.')
scoring = ['accuracy', 'precision', 'recall']           #define the metrics to evaluate the cross validation
start_time = time.time()         #start the counter
cv_scores = cross_validate(rf_clf, X, Y, cv=5, scoring=scoring, return_train_score=False)       #run 5 iterations of k-fold cross validation
elapsed_time = time.time() - start_time         #get the elapsed time since the counter was started
#print(sorted(cv_scores.keys()))
print(f"\nElapsed time to cross validate (5 Splits) the data with Random Forest ({dim_reduction} {n_features}) Classifier: {elapsed_time:.2f} seconds")
print("%0.2f accuracy with a standard deviation of %0.2f" % (cv_scores['test_accuracy'].mean()*100, cv_scores['test_accuracy'].std()))
print("%0.2f precision with a standard deviation of %0.2f" % (cv_scores['test_precision'].mean()*100, cv_scores['test_precision'].std()))
print("%0.2f recall with a standard deviation of %0.2f" % (cv_scores['test_recall'].mean()*100, cv_scores['test_recall'].std()))

# 5. Split train an validation dataset
print('\nDataset Split:')
X_train,X_valid,Y_train,Y_valid=train_test_split(X,Y,test_size=0.2,random_state=3)     #split the initial dataset into 80% training, 20% validation
print('train set: {}  | test set: {}'.format(round(((len(Y_train)*1.0)/len(X)),3),round((len(Y_valid)*1.0)/len(Y),3)))

# # 6. Hyperparameters Tuning
# print('\nHyperparameters Tuning Stage.\n')
# parameters = {'criterion':('gini', 'entropy'),'n_estimators':range(20,200,20), 'max_features':[1,2,3,4,5], 'max_depth':[1,2,3,4,5,6,7,8,9,10],'min_samples_split': [2,3,4,5]}   #define the values to change in all parameters
# grid_clf = GridSearchCV(estimator=rf_clf, param_grid=parameters, scoring=scoring, refit='accuracy')  #define the grid search classifier instance
# grid_clf.fit(X_train, Y_train)           #fit the data with all the parameters configurations
# tuning_scores = pd.DataFrame(grid_clf.cv_results_)       #store all the scrores in a dataframe
# #print(sorted(grid_clf.cv_results_.keys()))
# tuning_scores = tuning_scores[['rank_test_accuracy', 'param_criterion', 'param_n_estimators', 'param_max_features','param_max_depth', 'param_min_samples_split', 'mean_test_accuracy', 'mean_test_precision', 'mean_test_recall']]  # leave only the important scores
# tuning_scores['mean_test_accuracy'] = tuning_scores['mean_test_accuracy'].round(4)*100
# tuning_scores['mean_test_precision'] = tuning_scores['mean_test_precision'].round(4)*100
# tuning_scores['mean_test_recall'] = tuning_scores['mean_test_recall'].round(4)*100
# tuning_scores = tuning_scores.sort_values(by=['rank_test_accuracy'])    #rank the configurations by accuracy scores
# tuning_scores.to_csv('Tuning_Results_by_Accuracy.csv')  #save the configuration in a csv file

# 6. Train  Random Forest with best hyperparameters
print('\nTraining with Best Parameters')
criterion = 'entropy'
n_estimators = 140
max_features = 5
max_depth = 10
min_samples_split = 2
rf_clf_best = RandomForestClassifier(criterion=criterion, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split)    #create a new instance of the classifier with the best hyperparameters
rf_clf_best.fit(X_train,Y_train)        #Train the model using the whole initial set
elapsed_time = time.time() - start_time         #get the elapsed time since the counter was started

# 7. Plot the Learning Curve
# Cross validation with 100 iterations to get smoother mean test and trai score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
dt.plot_learning_curve(estimator=rf_clf_best, title="Learning Curve for RF with PCA (10) for best Accuracy", X=X, y=Y, cv=cv, n_jobs=4)
plt.show()

# 4. Get Performance Scores on Validation data
print('\nValidation with Best Parameters:')
Y_pred_RF = rf_clf_best.predict(X_valid)
RF_accuracy = round(accuracy_score(Y_valid,Y_pred_RF),2)*100     #get accuracy
RF_precision = round(precision_score(Y_valid,Y_pred_RF),2)*100       #get precision
RF_recall = round(recall_score(Y_valid,Y_pred_RF),2)*100               #get recall
print(f'\nRF ({dim_reduction} {n_features}) Accuracy Score on Validation data after tuning: {RF_accuracy}%')
print(f'RF ({dim_reduction} {n_features}) Precision Score on Validation data after tuning: {RF_precision}%')
print(f'RF ({dim_reduction} {n_features}) Recall Score on Validation data after tuning: {RF_recall}%')

# 8. Plot non-normalized confusion matrix for validation set
titles_options = [
    (f"Random Forest ({dim_reduction} {n_features}) Confusion Matrix (Valid)", None),
    #("{dim_reduction} Random Forest Normalized confusion matrix", "true"),       #In case we want the normalised matrix
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        rf_clf_best,
        X_valid,
        Y_valid,
        display_labels=["No Tumor", "Tumor"],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
plt.show()
#plt.savefig('Final_Valid_CM.png')

# 9. Get Prediction and Performance scores on Final Test dataset
print('\nTesting Stage.')
images_array, images_vectors_test, labels_test = dt.process_test_data()    #process test data
images_features_test = dt.reduce_dimensionality_with_PCA(images_vectors_test, n_features)    #project PCA features for test data
X_test, Y_test = images_features_test, labels_test
Y_pred_test = rf_clf_best.predict(X_test)      # predictions on test set
final_accuracy = round(accuracy_score(Y_test,Y_pred_test),3)*100
final_precision = round(precision_score(Y_test,Y_pred_test),3)*100
final_recall = round(recall_score(Y_test,Y_pred_test),3)*100
final_scores = pd.DataFrame([(final_accuracy, final_precision, final_recall)],columns=['Accuracy', 'Precision', 'Recall'])
final_scores.to_csv('Final_Test_Results.csv')    #save the final results in a csv file
print(f'Random Forest ({dim_reduction} {n_features}) Accuracy Score on Test data after tuning: {final_accuracy}%')
print(f'Random Forest ({dim_reduction} {n_features}) Precision Score on Test data after tuning: {final_precision}%')
print(f'Random Forest ({dim_reduction} {n_features}) Recall Score on Test data after tuning: {final_recall}%')

# 10. Plot non-normalized confusion matrix
titles_options = [
    (f"Random Forest ({dim_reduction} {n_features}) Confusion Matrix (Test)", None),
    #("{dim_reduction} Random Forest Normalized confusion matrix", "true"),       #In case we want the normalised matrix
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        rf_clf_best,
        X_test,
        Y_test,
        display_labels=["No Tumor", "Tumor"],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
plt.show()
#plt.savefig('Final_Test_CM.png')

# # 11. Random Forest  visualisation
# for index in range(0, 5):
#    classf.visualise_tree(rf_clf_best.estimators_[index])      #visualise 5 of the  Trees structures of the random forest
