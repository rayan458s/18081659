
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
images_features = dt.process_ANOVA_features(10)

#Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(images_features,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}\n'.format(round(((len(Y_train)*1.0)/len(images_features)),3),round((len(Y_test)*1.0)/len(labels),3)))

#Plot the features importances
forest_importances, std = dt.get_features_importance_with_RF(X_train, Y_train)
fig, ax = plt.subplots()            #define the plot object
forest_importances.plot.bar(yerr=std, ax=ax)        #plot ar graph
ax.set_title("ANOVA Feature importances using MDI")       #set title
ax.set_ylabel("Mean decrease in impurity")      #set y-label
fig.tight_layout()
plt.show()

########################################## DT CLASSIFIER ######################################################

# 1. Fit Decision Three model
tree_params={'criterion':'entropy'}
start_time = time.time()
Y_pred_DT, dt_clf = dt.Decision_Tree_Classifier(X_train, Y_train, X_test, tree_params)
elapsed_time = time.time() - start_time
print(f"Elapsed time to classify the data using Decision Three (ANOVA) Classifier: {elapsed_time/60:.2f} minutes")

# 2. Get Performance Scores
DT_accuracy = round(accuracy_score(Y_test,Y_pred_DT),2)*100     #get accuracy
DT_precision = round(precision_score(Y_test,Y_pred_DT),2)*100       #get precision
DT_recall = round(recall_score(Y_test,Y_pred_DT),2)*100
print('\nDecision Tree (ANOVA) Accuracy Score on Test data: {}%'.format(DT_accuracy))
print('\nDecision Tree (ANOVA) Precision Score on Test data: {}%'.format(DT_precision))
print('\nDecision Tree (ANOVA) Recall Score on Test data: {}%'.format(DT_recall))

# # 3. Decision Three visualisation
# dt.visualise_tree(dt_clf)

# 4. Plot non-normalized confusion matrix
titles_options = [
    ("Decision Three (ANOVA) Confusion Matrix", None),
    #("Decision Three (ANOVA) Normalized confusion matrix", "true"),
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
tree_params = {'criterion': 'entropy', 'min_samples_split':50}
start_time = time.time()
Y_pred_DT2, dt_clf_2 = dt.Decision_Tree_Classifier(X_train, Y_train, X_test, tree_params)
elapsed_time = time.time() - start_time
print(f"\nElapsed time to classify the data using Decision Three (ANOVA) Classifier after hyperparameters tuning: {elapsed_time/60:.2f} minutes")

# 6. Get Performance Scores
DT2_accuracy = round(accuracy_score(Y_test,Y_pred_DT2),2)*100     #get accuracy
DT2_precision = round(precision_score(Y_test,Y_pred_DT2),2)*100       #get precision
DT2_recall = round(recall_score(Y_test,Y_pred_DT2),2)*100
print('\nDecision (ANOVA) Tree (after tuning) Accuracy Score on Test data: {}%'.format(DT2_accuracy))
print('\nDecision (ANOVA) Tree (after tuning) Precision Score on Test data: {}%'.format(DT2_precision))
print('\nDecision (ANOVA) Tree (after tuning) Recall Score on Test data: {}%'.format(DT2_recall))


# 7. Plot non-normalized confusion matrix after tuning
titles_options = [
    ("Decision Three (ANOVA) Confusion Matrix After Tuning", None),
    #("Decision Three (ANOVA) Normalized confusion matrix", "true"),
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
print(f"\nElapsed time to classify the data using Random Forest (ANOVA) Classifier: {elapsed_time/60:.2f} minutes")

# 2. Get Performance Scores
RF_accuracy = round(accuracy_score(Y_test,Y_pred_RF),2)*100     #get accuracy
RF_precision = round(precision_score(Y_test,Y_pred_RF),2)*100       #get precision
RF_recall = round(recall_score(Y_test,Y_pred_RF),2)*100
print('\nRandom Forest (ANOVA) Accuracy Score on Test data: {}%'.format(RF_accuracy))
print('\nRandom Forest (ANOVA) Precision Score on Test data: {}%'.format(RF_accuracy))
print('\nRandom Forest (ANOVA) Recall Score on Test data: {}%'.format(RF_accuracy))

# # 3. Random Forest  visualisation
# for index in range(0, 5):
#     dt.visualise_tree(rf_clf.estimators_[index])

# 4. Plot non-normalized confusion matrix
titles_options = [
    ("ANOVA Random Forest Confusion Matrix", None),
    #("ANOVA Random Forest Normalized confusion matrix", "true"),
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

# 5. Remove unimportant features + retrain and re-visualise