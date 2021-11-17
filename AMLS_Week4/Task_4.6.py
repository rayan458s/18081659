
print("1. Importing libraries\n")
#Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#%%

#Loading the iris data
iris = load_iris()
print("2. IRIS dataset information:")
print('Features in the data: ', iris.feature_names)
print('Classes to predict: ', iris.target_names)

# For more details about this dataset, uncomment code below:
# print(data.DESCR)

#Extracting features
X = iris.data
### Extracting target/ class labels
y = iris.target

print('Number of examples in the data:', X.shape[0],'\n')

#Using the train_test_split to create train(75%) and test(25%) sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 47, test_size = 0.25)

print("3. One sample example: ")
# visualize one example in training set
print(f'E.x, A training example with feature: {X_train[0]} belongs to Class {y_train[0]}\n')


#Importing the Decision tree classifier from the sklearn library.
tree_params={'criterion':'entropy'}
clf = tree.DecisionTreeClassifier( **tree_params )

#Training the decision tree classifier on training set.
# Please complete the code below.
clf.fit(X_train,y_train)


#Predicting labels on the test set.
# Please complete the code below.
y_pred =  clf.predict(X_test)

print("4. Fitted Sample example: ")
print(f'Test feature {X_test[0]}\n True class {y_test[0]}\n predict class {y_pred[0]}\n')


print("5.Accuracy Evaluation: ")
#Use accuracy metric from sklearn.metrics library
print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))
print('Accuracy Score on test data: ', round(accuracy_score(y_true=y_test, y_pred=y_pred),2)*100,'%\n')


print('6.1 Visualise Tree Structure:\n')
def visualise_tree(tree_to_print):
    plt.figure()
    tree.plot_tree(tree_to_print,
               feature_names = iris.feature_names,
               class_names=iris.target_names,
               filled = True,
              rounded=True);
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=800)
    plt.show()

# def visualise_tree(tree_to_print):
#     dot_data = tree.export_graphviz(tree_to_print, out_file=None,
#                       feature_names=iris.feature_names,
#                       class_names=iris.target_names,
#                       filled=True, rounded=True,
#                       special_characters=True)
#     graph = ghz.Source(dot_data)
#     return graph

visualise_tree(clf)

print('6.2 Visualise Decision Boundary:\n')

def visualise_decision_boundary(**tree_params):
    # Parameters
    n_classes = 3
    plot_colors = "ryb"
    plot_step = 0.02
    plt.figure(figsize=(10,5),dpi=100)
    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
        # We only take the two corresponding features
        X_visualise = X[:, pair]
        y_visualise = y

        # Train
        clf = tree.DecisionTreeClassifier(**tree_params).fit(X_visualise, y_visualise)

        # Plot the decision boundary
        plt.subplot(2, 3, pairidx + 1)

        x_min, x_max = X_visualise[:, 0].min() - 1, X_visualise[:, 0].max() + 1
        y_min, y_max = X_visualise[:, 1].min() - 1, X_visualise[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

        plt.xlabel(iris.feature_names[pair[0]])
        plt.ylabel(iris.feature_names[pair[1]])

        # Plot the training points
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X_visualise[idx, 0], X_visualise[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

    plt.suptitle("Decision surface of a decision tree using paired features")
    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")
    plt.show()

visualise_decision_boundary(**tree_params)


print('7. Hyperparameter Tuning:')

tree_params={
    'criterion': 'entropy',
    'min_samples_split':50
}
clf = tree.DecisionTreeClassifier(**tree_params)
clf.fit(X_train, y_train)
print('Accuracy Score on train data: ', round(accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)),2)*100)
print('Accuracy Score on the test data: ', round(accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)),2)*100,'\n')

print('7.1 Visualise New Tree Structure:\n')
visualise_tree(clf)
print('7.2 Visualise New Decision Boundary:\n')
visualise_decision_boundary(**tree_params)



print('##########################  RANDOM FORESTS   ###############################')


print('1. Train and test')

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets
# Complete the code below.
clf.fit(X_train,y_train)

# prediction on test set
# complete code below.
y_pred=clf.predict(X_test)

print(f'Test feature {X_test[0]}\n True class {y_test[0]}\n predict class {y_pred[0]}\n')

print('2. Evaluation')
print("Random Forest test Accuracy:", round(accuracy_score(y_test, y_pred),2)*100,'%\n')

print('3. Random Forest Visualisation\n')
for index in range(0, 5):
    visualise_tree(clf.estimators_[index])

print('4. Identify Important features\n')

feature_name= iris.feature_names
feature_importance=clf.feature_importances_
plt.figure()
plt.bar(feature_name,feature_importance)
plt.xticks(rotation=45)
plt.ylabel('feature importance')
plt.show()

print('5. Drop un-important feature\n')
# drop least important feature
X=iris.data[:,[0,2,3]]

# split data again
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 47, test_size = 0.25)


print('6. Re-create the random forest\n')
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets
# complete code below.
clf.fit(X_train,y_train)

# prediction on test set
# complete code below.
y_pred=clf.predict(X_test)

print('7. Evaluation after removal')
# Model Accuracy, how often is the classifier correct?
print("After removing sepal width, Random Forest Accuracy:",round(accuracy_score(y_test, y_pred),2)*100,'%')




