import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Loading the CSV file
dataset=pandas.read_csv('datasets/multi_regr_data.csv')
print(dataset.shape) #(data_number,feature_number)

# Split the data, we will use first 2 columns as features and the 3rd columns as target.
X = dataset[list(dataset.columns)[:-1]]
#print(X.shape)
Y = dataset[list(dataset.columns)[-1]]
#print(Y.shape)
# Split the data into training and testing(75% training and 25% testing data)
xtrain,xtest,ytrain,ytest=train_test_split(X, Y, random_state=0)
print(xtrain.shape)
print(xtest.shape)

######################## 1 ##################

# sklearn functions implementation
def multilinearRegrPredict(xtrain, ytrain,xtest ):
    # Create linear regression object
    reg=LinearRegression()
    # Train the model using the training sets
    reg.fit(xtrain,ytrain)
    # Make predictions using the testing set
    y_pred = reg.predict(xtest)
    # See how good it works in test data,
    # we print out one of the true target and its estimate
    print('For the true target: ',list(ytest)[-1])
    print('We predict as: ', list(y_pred)[-1]) # print out the
    print("Overall Accuracy Score from library implementation:", reg.score(xtest, ytest)) #.score(Predicted value, Y axis of Test data) methods returns the Accuracy Score or how much percentage the predicted value and the actual value matches

    return y_pred

y_pred = multilinearRegrPredict(xtrain, ytrain, xtest )


######################## 2 ##################

def multiLinparamEstimates(xtrain, ytrain):
    # Q: why need 'intercept'?
    intercept = np.ones((xtrain.shape[0], 1))  #FIRST COLUMN OF THE X MATRIX
    print(xtrain.shape)
    xtrain = np.concatenate((intercept, xtrain), axis=1)
    print(xtrain.shape)

    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(xtrain), xtrain)),
                               np.transpose(xtrain)), ytrain)
    return beta

def multilinearNEWRegrPredict(xtrain, ytrain,xtest):
    beta = multiLinparamEstimates(xtrain, ytrain)
    print(beta)
    intercept = np.ones((xtest.shape[0], 1))  #FIRST COLUMN OF THE X MATRIX
    xtest = np.concatenate((intercept, xtest), axis=1)
    y_pred = np.matmul(xtest, beta)
    return y_pred


# Model Evaluation - R2 Score
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print("Accuracy Score from scratch implementation:", r2)
    return r2

y_pred1 = multilinearNEWRegrPredict(np.array(xtrain.values), np.array(ytrain.values).flatten(),
                             np.array(xtest.values))
#print (y_pred1)
r2=r2_score(ytest, y_pred1)


def SSR( y_pred,yTest):
    # Complete your code here.
    ssr = (1/(len(yTest)))*np.sum(np.subtract(yTest, y_pred)**2)
    return ssr

y_pred_SSR = SSR(y_pred, np.array(ytest.values).flatten())
#print(y_pred.shape)
#print(np.array(ytest.values).flatten().shape)
y_pred1_SSR = SSR(y_pred1, np.array(ytest.values).flatten())

print("Scikit-learn multivariate linear regression SSR: %.4f" % y_pred_SSR)
print("From scratch implementation of multivariate linear regression SSR: %.4f" % y_pred1_SSR)
