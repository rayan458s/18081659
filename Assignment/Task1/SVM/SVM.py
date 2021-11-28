
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import time

from Assignment.Package import data_processing as dt


########################################## DATA PROCESSING ######################################################

images_vectors, labels = dt.process_data()      #process all the images into pixel vectors and get the labels
#WARNING ONLY 5 AND 10 FEATURES MODES HAVE BEEN PROCESSED FOR ANOVA and PCA, IF YOU WISH TO USE A DIFFERENT NUMBER OF
#FEATURES YOU NEED TO FIRST USE THE dt.process_ANOVA_features(n_features) or dt.process_PCA_features(n_features) functions
n_features = 5     #define the number of features to select/project from the pixels vectors
dim_reduction = "ANOVA"
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


########################################## SVM CLASSIFIER ######################################################

#1. Test SVM model accuracy for different Kernel
kernels = ["linear", "rbf", "poly"]
SVM_scores_df = pd.DataFrame(kernels, columns=["kernels"])
accuracies = []
precisions = []
recalls = []
for kernel in kernels:
    Y_pred_SVM, svm_clf = dt.SVM_Classifier(X_train, Y_train, X_test,kernel)
    accuracies.append(round(metrics.accuracy_score(Y_test,Y_pred_SVM),2)*100)
    precisions.append(round(metrics.precision_score(Y_test,Y_pred_SVM),2)*100)
    recalls.append(round(metrics.recall_score(Y_test,Y_pred_SVM),2)*100)

SVM_scores_df['accuracies']=accuracies
SVM_scores_df['precisions']=precisions
SVM_scores_df['recalls']=recalls
print(f"\nSVM ({dim_reduction} {n_features}) performance:\n")
print(SVM_scores_df)

#2. Plot accuracy vs kernel
fig, ax = plt.subplots()
ax.scatter(SVM_scores_df['kernels'], SVM_scores_df['accuracies'], c='b', label='Accuracy')
ax.scatter(SVM_scores_df['kernels'], SVM_scores_df['precisions'], c='g', label='Precision')
ax.scatter(SVM_scores_df['kernels'], SVM_scores_df['recalls'], c='r', label='Recall')
ax.set(ylabel='Performance (%)',xlabel='Type of Kernel', ylim=[50, 110])
plt.legend(loc='lower right')
plt.grid(visible = True)
plt.title(f'SVM ({dim_reduction} {n_features}) Performance vs Type of Kernel', weight = 'bold')
plt.show()

# 3. Fit SVM model for rbf kernel
Kernel = "rbf"
start_time = time.time()        #start the time counter (to determine the time taken to classify the data
Y_pred_SVM, svm_clf = dt.SVM_Classifier(X_train, Y_train, X_test,Kernel)         #classify the data using the SVM Classifier
elapsed_time = time.time() - start_time     #get the elapsed time since the counter was started
print(f"\nElapsed time to classify the data using SVM ({dim_reduction} {n_features}) Classifier for Kernel = {Kernel} {elapsed_time:.2f} seconds")

# 4. Get Performance Scores
SVM_accuracy = round(accuracy_score(Y_test,Y_pred_SVM),2)*100     #get accuracy
SVM_precision = round(precision_score(Y_test,Y_pred_SVM),2)*100       #get precision
SVM_recall = round(recall_score(Y_test,Y_pred_SVM),2)*100               #get recall
print(f'\nSVM ({dim_reduction} {n_features}) Accuracy Score on Test data: {SVM_accuracy}%')
print(f'\nSVM ({dim_reduction} {n_features}) Precision Score on Test data: {SVM_precision}%')
print(f'\nSVM ({dim_reduction} {n_features}) Recall Score on Test data: {SVM_recall}%')

# 5. Plot non-normalized confusion matrix
titles_options = [
    (f"SVM ({dim_reduction} {n_features}) Confusion Matrix for {Kernel} Kernel", None),
    #("SVM Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        svm_clf,
        X_test,
        Y_test,
        display_labels=["No Tumor", "Tumor"],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
plt.show()