
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
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
print("\nSVM (PCA) performance:\n")
print(SVM_scores_df)

#2. Plot accuracy vs kernel
fig, ax = plt.subplots()
ax.scatter(SVM_scores_df['kernels'], SVM_scores_df['accuracies'], c='b', label='Accuracy')
ax.scatter(SVM_scores_df['kernels'], SVM_scores_df['precisions'], c='g', label='Precision')
ax.scatter(SVM_scores_df['kernels'], SVM_scores_df['recalls'], c='r', label='Recall')
ax.set(title = 'SVM (PCA) Performance vs Type of Kernel',
        ylabel='Performance (%)',xlabel='Type of Kernel', ylim=[50, 110])
plt.legend(loc='lower right')
plt.grid(visible = True)
plt.title('SVM (PCA) Performance vs Type of Kernel', weight = 'bold')
plt.show()

# 3. Fit SVM model for linear Kernel and get accuracy score
Y_pred_SVM, svm_clf = dt.SVM_Classifier(X_train, Y_train, X_test,"rbf")
print('\nSVM (PCA) Accuracy Score on Test data for rbf Kernel: {}\n'.format(round(metrics.accuracy_score(Y_test,Y_pred_SVM),3)*100))

# 4. Plot non-normalized confusion matrix
titles_options = [
    ("SVM (PCA) Confusion Matrix for rbf Kernel", None),
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