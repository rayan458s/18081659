# Date: 17/12/2021
# Institution: University College London
# Dept: EEE
# Class: ELEC0134
# SN: 18081659
# Description: Contains all the non-deep learning classifiers sued in this project


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt


def KNN_Classifier(X_train, Y_train, X_test,k):
    '''
    Input:
    - 'X_train': Training Features
    - 'Y_train': Training labels
    - 'X_test': Testing Features
    - 'k': Number of Nearest Neighbors
    Returns:
    - 'Y_pred_KNN': The predictions on the Testing Features
    - 'KNN_clf': The trained KNN  classifier
    '''
    KNN_clf = KNeighborsClassifier(n_neighbors=k)     #Create KNN object with a K coefficient
    KNN_clf.fit(X_train, Y_train)       # Fit KNN model
    Y_pred_KNN = KNN_clf.predict(X_test)       #Get predictions
    return Y_pred_KNN, KNN_clf


def Decision_Tree_Classifier(X_train, Y_train, X_test, tree_params):
    '''
    Input:
    - 'X_train': Training Features
    - 'Y_train': Training labels
    - 'X_test': Testing Features
    - 'tree_params': The Decision Tree Parameters
    Returns:
    - 'Y_pred_DT': The predictions on the Testing Features
    - 'dt_clf': The trained decision Tree classifier
    '''
    dt_clf = tree.DecisionTreeClassifier( **tree_params )      #Define the decision three classifier
    dt_clf.fit(X_train,Y_train)         # Fit DT model
    Y_pred_DT =  dt_clf.predict(X_test)         #Get predictions
    return Y_pred_DT, dt_clf


def Random_Forest_Classifier(X_train, Y_train, X_test, n_estimators):
    '''
    Input:
    - 'X_train': Training Features
    - 'Y_train': Training labels
    - 'X_test': Testing Features
    - 'n_estimators': Number of Trees in the forest
    Returns:
    - 'Y_pred_RF': The predictions on the Testing Features
    - 'rf_clf': The trained Random Forest  classifier
    '''
    rf_clf = RandomForestClassifier(n_estimators=n_estimators)   #Define the random forest classifier
    rf_clf.fit(X_train,Y_train)        # Fit RF model
    Y_pred_RF = rf_clf.predict(X_test)      # get predictions
    return Y_pred_RF, rf_clf


def visualise_tree(tree_to_print):
    '''
    Function to plot the tree in a DT or trees in a RF
    Input:
    - 'tree_to_print': trees
    '''
    plt.figure()
    tree.plot_tree(tree_to_print,filled = True,rounded=True);
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=800)
    plt.show()


def Bagging_Classifier(X_train, y_train, X_test, k):
    '''
    Input:
    - 'X_train': Training Features
    - 'Y_train': Training labels
    - 'X_test': Testing Features
    - 'k': Number of Nearest Neighbors
    Returns:
    - 'Y_pred_BAG': The predictions on the Testing Features
    - 'bag_clf': The trained KNN  classifier
    '''
    bag_clf = BaggingClassifier(n_estimators=k,max_samples=0.5, max_features=4,random_state=1)  #Create KNN object with a K coefficient
    bag_clf.fit(X_train, y_train)      #Fit KNN model
    Y_pred_BAG = bag_clf.predict(X_test)    #get predictions
    return Y_pred_BAG, bag_clf


def Boosting_Classifier(X_train, Y_train, X_test, estimator, k):
    '''
    Input:
    - 'X_train': Training Features
    - 'Y_train': Training labels
    - 'X_test': Testing Features
    - 'k': The Decision Tree Parameter
    Returns:
    - 'Y_pred_BOOST': The predictions on the Testing Features
    - 'boost_clf': The trained decision Boosting classifier
    '''
    boost_clf = AdaBoostClassifier(base_estimator=estimator, n_estimators=k)       # AdaBoost takes Decision Tree as its base-estimator model by default.
    boost_clf.fit(X_train,Y_train,sample_weight=None)  # Fit DT boosted model
    Y_pred_BOOST = boost_clf.predict(X_test)    #get predictions
    return Y_pred_BOOST, boost_clf


def SVM_Classifier(X_train,Y_train, X_test, kernel):
    '''
    Input:
    - 'X_train': Training Features
    - 'Y_train': Training labels
    - 'X_test': Testing Features
    - 'kernel': SVM kernel function
    Returns:
    - 'y_pred': The predictions on the Testing Features
    - 'svm_clf': The trained SVM classifier
    '''
    svm_clf = SVC(kernel=kernel)        #create SVM classifier instance
    svm_clf.fit(X_train,Y_train)        #fit the training data
    y_pred = svm_clf.predict(X_test)    #get predictions
    return y_pred, svm_clf
