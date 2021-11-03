import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patches
from scipy.special import expit
import itertools

# Loading the TXT file
fruits = pd.read_table('datasets/classification_data.txt')


# Split the data
feature_names = ['mass', 'width', 'height', 'color_score']
x = fruits[feature_names]
y = fruits['fruit_label']
#print(x)
# Split the data into training and testing(75% training and 25% testing data)
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=0)
#print(x_train)
# Pre-process data
scaler = MinMaxScaler() # This estimator scales and translates each feature individually such that it is in the given range on the training set, default between(0,1)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#print(x_train)


# sklearn functions implementation
def logRegrPredict(x_train, y_train, xtest):
    # Build Logistic Regression Model
    logreg = LogisticRegression(solver='lbfgs')
    # Train the model using the training sets
    logreg.fit(x_train, y_train)
    y_pred= logreg.predict(xtest)
    #print('Accuracy on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
    return y_pred

y_pred = logRegrPredict(x_train, y_train,x_test)
#print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
#print(classification_report(y_test,y_pred))#text report showing the main classification metrics



#%%

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def logRegParamEstimates(xTrain, yTrain):
    intercept = np.ones((xTrain.shape[0], 1))
    xTrain = np.concatenate((intercept, xTrain), axis=1)
    yTrain[yTrain > 1] = 0
    theta = np.zeros(xTrain.shape[1])
    yTrain = yTrain.reset_index(drop=True)
    #print(theta)
    #print(xTrain)
    #print(yTrain)
    #print(sigmoid(theta*xTrain[1])-yTrain[1])

    for i in range(100):
        z = np.dot(xTrain, theta)
        h = sigmoid(z)
        lr = 0.01
        #print(loss(h,yTrain))
        '''
        gradient = 0
        for j in range(len(yTrain)):
            #print(sigmoid(np.dot(theta,xTrain[j]))-yTrain[j])
            gradient = gradient + (sigmoid(np.dot(theta,xTrain[j]))-yTrain[j])*xTrain[j]
        gradient = gradient * 1/(np.log(2))
        '''
        #gradient = loss(h,yTrain)
        print(np.log(2))
        gradient = (1/(np.log(2))) * np.dot(xTrain, (h-yTrain)).mean()
        theta = theta - lr * gradient
    return theta

def logRegrNEWRegrPredict(xTrain, yTrain, xTest ):
    theta = logRegParamEstimates(xTrain, yTrain)
    intercept = np.ones((xTest.shape[0], 1))
    xTest = np.concatenate((intercept, xTest), axis=1)
    sig = sigmoid(np.dot(xTest, theta))
    y_pred1 = sig
    return y_pred1


#%%

y_pred1 = logRegrNEWRegrPredict(x_train, y_train,x_test)
print(y_test)
print (y_pred1)
#print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred1)))
#print(classification_report(y_test,y_pred1))#text report showing the main classification metrics


