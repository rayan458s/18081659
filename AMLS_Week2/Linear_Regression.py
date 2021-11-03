import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Loading the CSV file
houseprice=pandas.read_csv('datasets/regression_data.csv')
houseprice=houseprice[['Price (Older)', 'Price (New)']] # Choose 2 columns


# Split the data
X=houseprice[['Price (Older)']]
Y=houseprice[['Price (New)']]

# Split the data into training and testing(75% training and 25% testing data)
xTrain,xTest,yTrain,yTest = train_test_split(X,Y)
#print(X)
#print(xTrain)

############################# PART 1 #############################

# sklearn functions implementation
def linearRegrPredict(xTrain, yTrain,xTest ):
    # Create linear regression object
    regr=LinearRegression()
    # Train the model using the training sets
    regr.fit(xTrain,yTrain)
    # Make predictions using the testing set
    y_pred = regr.predict(xTest)
    print("Accuracy Score:", regr.score(xTest,yTest)*100)  #Getting the percentage of correct predictions
    return y_pred


y_pred = linearRegrPredict(xTrain, yTrain, xTest)


# Plot testing set predictions
plt.scatter(xTest, yTest)
plt.plot(xTest, y_pred, 'r-')
plt.show()


############################# PART 2 #############################


# Transform dataframes to numpy arrays
xTrain1 = np.array(xTrain.values).flatten()
xTest1 = np.array(xTest.values).flatten()
yTrain1 = np.array(yTrain.values).flatten()
yTest1 = np.array(yTest.values).flatten()


def paramEstimates(xTrain, yTrain):
    beta = np.sum(np.multiply(xTrain,(np.add(yTrain, -np.mean(yTrain))))) / np.sum(np.multiply(xTrain, (np.add(xTrain, - np.mean(xTrain)))))
    xbar = (1/len(xTrain))*np.sum(xTrain)
    ybar = (1/len(yTrain))*np.sum(yTrain)
    alpha = ybar-beta*xbar

    return alpha, beta


def linearRegrNEWPredict(xTrain, yTrain,xTest):
    alpha, beta = paramEstimates(xTrain, yTrain)
    print(alpha)
    print(beta)
    # Complete the code here.

    y_pred1 = alpha + beta*xTest
    #y_pred1 = np.add(alpha, np.multiply(beta,xTest))
    return y_pred1


y_pred1=linearRegrNEWPredict(xTrain1, yTrain1,xTest1)


#Plot testing set predictions
plt.scatter(xTest, yTest)
plt.plot(xTest1, y_pred1, 'r-')
plt.show()


############################# PART 3 #############################


def SSR(yTest, y_pred):
    ssr = (1/(len(yTest)))*np.sum(np.subtract(yTest, y_pred)**2)
    return ssr

y_pred_SSR = SSR(yTest,y_pred)
y_pred1_SSR = SSR(yTest1,y_pred1)

print("Scikit-learn linear regression SSR: %.4f" % y_pred_SSR)
print("Our implementation of linear regression SSR: %.4f" % y_pred1_SSR)
