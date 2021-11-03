
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
print(boston.data.shape) # shape of data
#print(boston.feature_names)
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names) # convert the boston.data to a a dataframe
boston_df['Price']=boston.target # there is no column called ‘PRICE’ in the data frame because the target column is available in another attribute called target
newX=boston_df.drop('Price',axis=1) # All other features
newY=boston_df['Price'] # Boston Housing Price
boston_df.head()


X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.3,random_state=3)
#test_size= should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
#everytime you run it without specifying random_state, you will get a different result, this is expected behavior
#print (len(X_test), len(y_test))

print('train set: {}  | test set: {}'.format(round(len(y_train)/len(newX),2),
                                                       round(len(y_test)/len(newX),2)))

# generate a range of alpha values and put them in a numpy array
#r_alphas = 10**np.linspace(10,-2,100)*0.5
r_alphas = [0.001, 0.01, 0.1, 1, 10]
#print(r_alphas)

def ridgeRegrCVPredict(X_train, y_train, r_alphas,X_test):

    ridgecv = RidgeCV(alphas = r_alphas, fit_intercept=True)

    # Next step: fit ridgecv!
    #print('Best alpha value: '+str(ridgecv.alpha_))

    #Y_pred_cv = ...
    return Y_pred_cv
