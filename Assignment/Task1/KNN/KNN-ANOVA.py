
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from Assignment.Package import data_processing as dt

img_folder=r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/image'
label_file = r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/label.csv'


########################################## DATA PROCESSING ######################################################
#Get images (inputs) array
images_array, class_name = dt.load_images(img_folder)
images_array = np.array(images_array)

print("\nDataset shape: {}".format(images_array.shape))
a,b,c,d = images_array.shape
print("\nImage Size: {}x{}x{}".format(b,c,d))
print("\nNumber of Images: {}".format(a))

#Get labels (outputs) array
labels = dt.load_labels(label_file)
#print(labels)
print("\nNumber of Labels: {}".format(len(labels)))

#Array to  Vectors
images_vectors = dt.image_array_to_vector(images_array)
print("\nVector Size: {}".format(len(images_vectors[0])))

#Select 10 Features using ANOVA
k_ANOVA = 10
start_time = time.time()
images_features = dt.select_features_with_ANOVA(images_vectors, labels, k_ANOVA)
elapsed_time = time.time() - start_time
print(f"\nElapsed time to select features using ANOVA: {elapsed_time/60:.2f} minutes")
print("\nSelected number of features: {}".format(k_ANOVA))
print("\nFinal Input Data Shape: {}".format(np.array(images_features).shape))

#Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(images_features,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}\n'.format(round(((len(Y_train)*1.0)/len(images_features)),3),round((len(Y_test)*1.0)/len(labels),3)))

#Plot the features importances
forest_importances, std = dt.get_features_importance_with_RF(X_train, Y_train)
fig, ax = plt.subplots()            #define the plot object
forest_importances.plot.bar(yerr=std, ax=ax)        #plot ar graph
ax.set_title("KNN with ANOVA Feature importances using MDI")       #set title
ax.set_ylabel("Mean decrease in impurity")      #set y-label
fig.tight_layout()
plt.show()

########################################## KNN WITH ANOVA CLASSIFIER ######################################################

#1. Fit KNN model for different values of k
estimators = [1,20]
accuracies_df = pd.DataFrame(list(range(1,estimators[1])), columns=["k"])
accuracies = []
for i in range(estimators[0],estimators[1]):
    Y_pred_KNN, KNN_clf = dt.KNN_Classifier(X_train, Y_train, X_test,i)
    accuracies.append(round(accuracy_score(Y_test,Y_pred_KNN),3)*100)
    print('Accuracy for k={} computed'.format(i))

accuracies_df['accuracies']=accuracies
print('\nKNN with ANOVA Accuracy Score on Test data for different values of K:\n',accuracies_df)
print('\nThe value of K should be below {}'.format(round(np.sqrt(len(X_train)))))

#2. Plot accuracy vs k
fig, ax = plt.subplots()
ax.scatter(accuracies_df['k'], accuracies_df['accuracies'])
ax.set(title = 'Accuracy against number of neighbors K',
            ylabel='Accuracy (%)',xlabel='Number of Estimators K', ylim=[60, 100])
plt.title('Accuracy vs number of neighbors K for KNN with ANOVA', weight = 'bold')
ax.xaxis.set_ticks(np.arange(0, 21, 2))
plt.grid(visible = True)
plt.show()

# 3. Fit KNN model for K = 18 and get accuracy score
start_time = time.time()
Y_pred, KNN_clf = dt.KNN_Classifier(X_train, Y_train, X_test,18)
print('\nKNN with ANOVA Accuracy Score on Test data: {}%\n'.format(round(metrics.accuracy_score(Y_test,Y_pred),3)*100))
elapsed_time = time.time() - start_time
print(f"Elapsed time to classify the data using KNN with ANOVA: {elapsed_time/60:.2f} minutes")

# 4. Plot non-normalized confusion matrix
titles_options = [
    ("KNN with ANOVA Confusion matrix", None),
    #("KNN Normalized confusion matrix", "true"),
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