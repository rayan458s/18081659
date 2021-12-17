# Date: 17/12/2021
# Institution: University College London
# Dept: EEE
# Class: ELEC0134
# SN: 18081659
# Description: Random Forest Testing. See plots file for Results

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import pyplot
import time
import pandas as pd

from Assignment.Functions import data_processing as dt
from Assignment.Functions import Classifiers as classf

########################################## DATA PROCESSING ######################################################

# 1. Get the input data (matrix of image vectors and output data (vector of labels)
images_array, images_vectors, labels = dt.process_multiclass_data()      #process all the images into pixel vectors and get the labels

# 2. Transform inputs into feature vectors
dim_reduction = 'PCA'       #define the DR method
n_features = 10     #define the number of features to select/project from the pixels vectors
filepath = '/Users/rayan/PycharmProjects/AMLS/Assignment/Task2/Features/'+dim_reduction+'-'+str(n_features)+'.dat'  #get corresponding filepath
images_features = dt.get_features(filepath)
X, Y = images_features, labels

# #Plot the features importances
# dt.plot_features_importances(X, Y, dim_reduction, n_features)

#Split train an test dataset
X_train,X_valid,Y_train,Y_valid=train_test_split(X,Y,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}\n'.format(round(((len(Y_train)*1.0)/len(X)),3),round((len(Y_valid)*1.0)/len(Y),3)))


########################################## RF CLASSIFIER ######################################################

#1. Number of Trees Tunning
#print(rf_clf.get_params())
RF_scores_df = pd.DataFrame(list(range(5,300,10)), columns=["T"])      #create pandas dataframe to containe the performance metrics for difference values of K
accuracies = []
estimators_range = [5,300]      #the range to test the number of nearest neighbors
for i in range(estimators_range[0],estimators_range[1],10):        #classify the data for every value of K and get accuracy, precision and recall metrics
    Y_pred_RF, rf_clf = classf.Random_Forest_Classifier(X_train, Y_train, X_valid,i)
    accuracies.append(round(accuracy_score(Y_valid,Y_pred_RF),2)*100)
RF_scores_df['accuracies']=accuracies      #create new column in the dataframe with the accuracies
print(f"\nRF ({dim_reduction} {n_features}) accuracy:\n")
print(RF_scores_df)

#2. Accuracy Vs Number of Trees Visualisation
# visualise the performance metrics vs the number of trees in a plot
fig, ax = plt.subplots()
ax.scatter(RF_scores_df['T'], RF_scores_df['accuracies'])
ax.set(ylabel='Performance (%)',xlabel='Number of Trees', ylim =[65,85])
plt.grid(visible = True)
plt.title(f'Random Forest ({dim_reduction} {n_features}) Accuracy vs Number of Trees', weight = 'bold')
plt.show()
# filename = f'T2-RF-{dim_reduction}-{n_features}-PERF'
# pyplot.savefig(filename + '.png')

# 3. Classify using Random Forest with best hyperparameter
n_trees = 245
start_time = time.time()        #start the counter
Y_pred_RF, rf_clf = classf.Random_Forest_Classifier(X_train, Y_train, X_valid,n_trees)
elapsed_time = time.time() - start_time         #get the elapsed time since the counter was started
print(f"\nElapsed time to classify the data with Random Forest ({dim_reduction} {n_features}) Classifier: {elapsed_time:.2f} seconds")
RF_accuracy = round(accuracy_score(Y_valid,Y_pred_RF),2)*100     #get accuracy
print(f'\nRF ({dim_reduction} {n_features}) Accuracy Score on Test data: {RF_accuracy}%')


# 4. Plot non-normalized confusion matrix
titles_options = [
    (f"Random Forest ({dim_reduction} {n_features}) Confusion Matrix", None),
    #("{dim_reduction} Random Forest Normalized confusion matrix", "true"),       #In case we want the normalised matrix
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        rf_clf,
        X_valid,
        Y_valid,
        display_labels=['glioma', 'meningioma', 'no_tumor', 'pituitary'],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
plt.show()



# # 5. Random Forest  visualisation
# for index in range(0, 5):
#    classf.visualise_tree(rf_clf.estimators_[index])      #visualise 5 of the  Trees structures of the random forest
