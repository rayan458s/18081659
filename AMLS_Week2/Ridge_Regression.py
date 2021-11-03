import matplotlib.pyplot as plt
import numpy as np
from pandas import *
import pandas as pd
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error


#add another column that contains the house prices which in scikit learn datasets are considered as target
boston=load_boston() # get the data
#print(boston.keys()) # boston variable itself is a dictionary, so we can check for its keys
#print(boston.data.shape) # shape of data
#print(boston.feature_names)
boston_df = pd.DataFrame(boston.data,columns=boston.feature_names) # convert the boston.data to a a dataframe
boston_df['Price'] = boston.target # there is no column called ‘PRICE’ in the data frame because the target column is available in another attribute called target
newX=boston_df.drop('Price',axis=1) # All other features
newY=boston_df['Price'] # Boston Housing Price
boston_df.head()

X_train,X_test,y_train,y_test = train_test_split(newX,newY,test_size=0.3,random_state=3)
#test_size= should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
#everytime you run it without specifying random_state, you will get a different result, this is expected behavior
#print (len(X_test), len(y_test))

print('train set: {}  | test set: {}'.format(round(len(y_train)/len(newX),2),
                                                       round(len(y_test)/len(newX),2)))

def ridgeRegr(X_train, y_train, X_test):

    #Create linear regression object with a ridge coefficient 0.1
    ridge_regr_model = Ridge(alpha=0.1,fit_intercept=True)
    ridge_regr_model.fit(X_train, y_train) # Fit Ridge regression model


    Y_pred = ridge_regr_model.predict(X_test)
    #print (Y_pred)
    return Y_pred

Y_pred=ridgeRegr(X_train, y_train, X_test)

plt.scatter(y_test, Y_pred)
plt.xlabel("Actual prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show() #Ideally, the scatter plot should create a linear line. Since the model does not fit 100%, the scatter plot is not creating a linear line.

def paramEstimate(X_train, y_train, alpha):
    n, d = X_train.shape
    I = np.identity(d)
    L = np.dot(X_train.transpose(), X_train) + np.dot(alpha, I)
    L_1 = np.linalg.inv(L)
    w_rr = L_1.dot(X_train.transpose()).dot(y_train)
    return w_rr

def ridgeRegrNEW(xTrain, yTrain, alpha,X_test):
    w_rr = paramEstimate(xTrain, yTrain, alpha)
    y_pred= np.matmul(X_test, w_rr)
    return y_pred



#%%

y_pred = ridgeRegrNEW(X_train, y_train, 0.1,X_test)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()

mse = mean_squared_error(y_test, Y_pred) # check the level of error of a model
print('Mean Squared Error (MSE) on test set (built-in model): '+str(mse))
mse2=mean_squared_error(y_test, y_pred)
print('Mean Squared Error (MSE) on test set (from scratch model): '+ str(mse2))


