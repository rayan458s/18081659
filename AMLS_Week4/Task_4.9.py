print("1. Importing libraries\n")
import matplotlib.pyplot as plt
import numpy as np
from pandas import *
import pandas as pd
from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics


print("2. Load Data\n")
irisData=load_iris() # get the data
print(irisData.data.shape) # shape of data: 150 data points and 4 features
print(irisData.feature_names)# Feature_names of data
irisData_df=pd.DataFrame(irisData.data,columns=irisData.feature_names) # convert the irisData.data to a a dataframe
irisData_df['Species']=irisData.target # there is no column called ‘Species’ in the data frame because the target column is available in another attribute called target
newX=irisData_df.drop('Species',axis=1) # All other features
newY=irisData_df['Species'] # Species types
irisData_df.head()


X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.3,random_state=3)
#test_size= should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
#everytime you run it without specifying random_state, you will get a different result, this is expected behavior
#print (len(X_test), len(y_test))

print('train set: {}  | test set: {}'.format(round(((len(y_train)*1.0)/len(newX)),3),
                                                       round((len(y_test)*1.0)/len(newX),3)))

print("\n3. Bagging Classifier\n")
def baggingClassifierML(X_train, y_train, X_test,k):

    #Create KNN object with a K coefficient
    bagmodel=BaggingClassifier(n_estimators=k,max_samples=0.5, max_features=4,random_state=1)
    bagmodel.fit(X_train, y_train) # Fit KNN model


    Y_pred = bagmodel.predict(X_test)
    #print (Y_pred)
    return Y_pred

print("4. Bagging Classifier Accuracy\n")
Y_pred=baggingClassifierML(X_train, y_train, X_test,2)
score=metrics.accuracy_score(y_test,Y_pred)
print(round(score*100,1),'%')



print("\n5. Boosting Classifier \n")

def boostingClassifierML(X_train, y_train, X_test,k):
    # AdaBoost takes Decision Tree as its base-estimator model by default.
    boostmodel=AdaBoostClassifier(n_estimators=k)
    boostmodel.fit(X_train,y_train,sample_weight=None) # Fit KNN model


    Y_pred = boostmodel.predict(X_test)
    #print (Y_pred)
    return Y_pred

print("6. Boosting Classifier Accuracy\n")

Y_pred1=boostingClassifierML(X_train, y_train, X_test, 2)
score1=metrics.accuracy_score(y_test,Y_pred)
print(round(score1*100,1),'%')

print("7.1 Tune number of estimators for BAGGING\n")


accuracies_df = pd.DataFrame(list(range(1,31)), columns=["k"])
accuracies = []
for i in range(1,31):
    Y_pred=baggingClassifierML(X_train, y_train, X_test,i)
    accuracies.append(round(metrics.accuracy_score(y_test,Y_pred),3)*100)

accuracies_df['accuracies']=accuracies

print(accuracies_df)


fig, ax = plt.subplots()
ax.scatter(accuracies_df['k'], accuracies_df['accuracies'])
ax.set(title = 'Accuracy against number of estimators',
        ylabel='Accuracy',xlabel='K', ylim=[92, 100])
plt.title('Accuracy against number of number of estimators', weight = 'bold')

plt.show()


print("7.1 Tune number of estimators for BOOSTING\n")


accuracies1_df = pd.DataFrame(list(range(1,31)), columns=["k"])
accuracies1 = []
for i in range(1,31):
    Y_pred=boostingClassifierML(X_train, y_train, X_test,i)
    accuracies1.append(round(metrics.accuracy_score(y_test,Y_pred),3)*100)

accuracies1_df['accuracies']=accuracies1

print(accuracies1_df)


fig, ax = plt.subplots()
ax.scatter(accuracies1_df['k'], accuracies1_df['accuracies'])
ax.set(title = 'Accuracy against number of estimators',
        ylabel='Accuracy',xlabel='K', ylim=[92, 100])
plt.title('Accuracy against number of number of estimators', weight = 'bold')

plt.show()


