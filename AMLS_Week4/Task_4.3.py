import matplotlib.pyplot as plt
import numpy as np
from pandas import *
import pandas as pd
from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics



#add another column that contains the Species which in scikit learn datasets are considered as target
irisData=load_iris() # get the data
print(irisData.data.shape) # shape of data: 150 data points and 4 features
print(irisData.feature_names)# Feature_names of data
irisData_df=pd.DataFrame(irisData.data,columns=irisData.feature_names) # convert the irisData.data to a a dataframe
irisData_df['Species']=irisData.target # there is no column called ‘Species’ in the data frame because the target column is available in another attribute called target
#print(irisData_df)
newX=irisData_df.drop('Species',axis=1) # All other features
newY=irisData_df['Species'] # Species types
irisData_df.head()

X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.3,random_state=3)
#test_size= should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
#everytime you run it without specifying random_state, you will get a different result, this is expected behavior
#print (len(X_test), len(y_test))

print('train set: {}  | test set: {}'.format(round(((len(y_train)*1.0)/len(newX)),3),round((len(y_test)*1.0)/len(newX),3)))

def KNNClassifier(X_train, y_train, X_test,k):

    #Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train) # Fit KNN model


    Y_pred = neigh.predict(X_test)
    return Y_pred


accuracies_df = pd.DataFrame(list(range(1,31)), columns=["k"])
accuracies = []
for i in range(1,31):
    Y_pred=KNNClassifier(X_train, y_train, X_test,i)
    accuracies.append(round(metrics.accuracy_score(y_test,Y_pred),3)*100)

accuracies_df['accuracies']=accuracies

print(accuracies_df)


fig, ax = plt.subplots()
ax.scatter(accuracies_df['k'], accuracies_df['accuracies'])
ax.set(title = 'Accuracy against number of neighbors K',
        ylabel='Accuracy',xlabel='K', ylim=[92, 100])
plt.title('Accuracy against number of neighbors K', weight = 'bold')


print('The value of K should be below {}'.format(round(np.sqrt(len(X_train)))))


plt.show()
